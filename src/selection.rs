//! Diversity-aware read selection.
//!
//! # Why this exists
//!
//! The chain does not score a miner on its best energy alone. A proof set is
//! accepted only when it clears three gates at once:
//!
//! * every solution's energy is below the target,
//! * at least `min_solutions` solutions are present, and
//! * the set's *diversity* — the mean pairwise flip-invariant Hamming
//!   distance over the set, normalised by the problem width — reaches
//!   `min_diversity_milli` (200 on the devnet).
//!
//! A set of N copies of the same state has a diversity of exactly zero, and
//! is rejected however low that state's energy is. That makes the *spread* of
//! the returned set a property of the solver, not only of the chain: a
//! solver that returns many near-identical reads has handed the chain a set
//! it cannot select a diverse proof from.
//!
//! # What this module does
//!
//! It samples more reads than the job asks for, then returns a subset that
//! is simultaneously low-energy and spread out. The selection keeps the
//! lowest-energy read and then adds, one at a time, the read whose distance
//! to the already-chosen set is largest — the same farthest-point rule the
//! chain uses in `quantum_validation::select_diverse`.
//!
//! The chain's own rule starts from the most distant *pair*, so it can leave
//! the single best read out of the set it selects. That is the right rule for
//! the chain, which picks a proof out of a miner's candidates, and the wrong
//! rule for a miner, which is paid on the quality of the answer it hands
//! over. A warm-started read — the one the coordinator planted a known ground
//! state in — is exactly the read that must survive.
//!
//! # Cost
//!
//! The oversample costs wall clock, not quality: the reads run one model per
//! core, so the extra reads fill the machine that is already busy. The
//! returned set is the same size the job asked for.

use std::sync::atomic::{AtomicUsize, Ordering};

use quip_protocol::scoring::hamming_flip_invariant;
use quip_solver_core::{SampleParams, SamplerResult};

/// The default oversample.
///
/// Two is the compromise between the two things the chain rewards. A proof
/// set is scored on its energy and on its diversity, and a set of
/// near-identical reads has a diversity of zero however low its energy is,
/// so some oversampling pays. But a proof-of-work round is a *throughput*
/// contest — the chain hands the miner a lease of salts and takes the best
/// answer it can produce before the round closes — and every extra read is a
/// salt the miner did not answer. An operator whose chain runs a diversity
/// floor of zero should turn the factor down; the `oversample` key in the
/// backend's configuration does that.
pub const DEFAULT_OVERSAMPLE: usize = 2;

/// The read count to sample for a job that asked for `num_reads`.
///
/// `factor` is the oversample: one means no oversampling at all.
#[must_use]
pub fn oversampled_reads_with(num_reads: usize, factor: usize) -> usize {
    num_reads.max(1) * factor.max(1)
}

/// The factor the process is currently oversampling by.
///
/// One miner runs in one process, so the factor is a process-wide setting
/// rather than a field on the sampler.
static OVERSAMPLE_FACTOR: AtomicUsize = AtomicUsize::new(DEFAULT_OVERSAMPLE);

/// Set the factor. A value of zero is read as one: no oversampling is a
/// legitimate choice, and a zero factor would ask for no reads at all.
pub fn set_oversample(factor: usize) {
    OVERSAMPLE_FACTOR.store(factor.max(1), Ordering::Relaxed);
}

/// The factor in force.
#[must_use]
pub fn oversample() -> usize {
    OVERSAMPLE_FACTOR.load(Ordering::Relaxed)
}

/// The read count to sample, at the default factor.
#[must_use]
pub fn oversampled_reads(num_reads: usize) -> usize {
    oversampled_reads_with(num_reads, oversample())
}

/// Replace `reads` with the best `target` of them, spread out.
///
/// The input is sorted by energy, lowest first. The returned set always
/// contains the best read, and the rest are added farthest-first.
pub fn select_best_diverse(reads: &mut Vec<SamplerResult>, target: usize) {
    if reads.is_empty() || target == 0 {
        reads.clear();
        return;
    }
    if reads.len() <= target {
        return;
    }

    // Lowest energy first.
    reads.sort_by(|a, b| a.energy_milli.cmp(&b.energy_milli));

    let mut chosen = vec![0usize];
    let mut min_dist: Vec<u32> = (1..reads.len())
        .map(|candidate| hamming_flip_invariant(&reads[candidate].spins, &reads[0].spins))
        .collect();

    while chosen.len() < target {
        let mut pick: Option<usize> = None;
        let mut pick_dist = 0u32;
        for (offset, &distance) in min_dist.iter().enumerate() {
            if pick.is_none() || distance > pick_dist {
                pick = Some(offset + 1);
                pick_dist = distance;
            }
        }
        let Some(index) = pick else { break };
        chosen.push(index);
        for (offset, slot) in min_dist.iter_mut().enumerate() {
            let candidate = offset + 1;
            let distance =
                hamming_flip_invariant(&reads[candidate].spins, &reads[index].spins);
            if distance < *slot {
                *slot = distance;
            }
        }
    }

    // Keep the chosen reads, in energy order.
    let mut selected: Vec<SamplerResult> = chosen
        .into_iter()
        .map(|index| reads[index].clone())
        .collect();
    selected.sort_by(|a, b| a.energy_milli.cmp(&b.energy_milli));
    *reads = selected;
}

/// The chain's diversity score for a set of reads, in milli.
///
/// This is `quip_protocol::diversity::diversity_milli` over the read spins,
/// spelled out here so a test can assert on the number the chain will see.
#[must_use]
pub fn diversity_milli(reads: &[SamplerResult]) -> u32 {
    let spins: Vec<&[i8]> = reads.iter().map(|r| r.spins.as_slice()).collect();
    quip_protocol::diversity::diversity_milli(&spins)
}

/// Sample an oversampled read count, then return the best `num_reads` of them.
///
/// `sample` is handed a copy of `params` with the read count raised; it
/// returns the reads it produced. The returned set is the size the job asked
/// for.
pub fn sample_oversampled<F>(params: &SampleParams, mut sample: F) -> Vec<SamplerResult>
where
    F: FnMut(&SampleParams) -> Vec<SamplerResult>,
{
    let oversampled = SampleParams {
        num_reads: oversampled_reads(params.num_reads),
        ..params.clone()
    };
    let mut results = sample(&oversampled);
    select_best_diverse(&mut results, params.num_reads.max(1));
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(spins: Vec<i8>, energy_milli: i64) -> SamplerResult {
        SamplerResult { spins, energy_milli }
    }

    #[test]
    fn the_best_read_is_always_kept() {
        let mut reads = vec![
            read(vec![1, 1, 1, 1], -100),
            read(vec![1, 1, 1, -1], -200),
            read(vec![-1, -1, -1, -1], -50),
        ];
        select_best_diverse(&mut reads, 2);
        assert_eq!(reads.len(), 2);
        assert_eq!(reads[0].energy_milli, -200);
    }

    #[test]
    fn the_selected_set_is_spread_out() {
        // Four reads: one at the bottom energy, three near-copies of each
        // other. The selection must not return three near-copies.
        let mut reads = vec![
            read(vec![1, 1, 1, 1, 1, 1, 1, 1], -400),
            read(vec![1, 1, 1, 1, -1, -1, -1, -1], -100),
            read(vec![1, 1, 1, 1, -1, -1, -1, -1], -100),
            read(vec![-1, -1, -1, -1, 1, 1, 1, 1], -100),
        ];
        select_best_diverse(&mut reads, 2);
        assert_eq!(reads.len(), 2);
        // The second pick must differ from the first.
        assert!(diversity_milli(&reads) > 0);
    }

    #[test]
    fn a_short_set_is_left_alone() {
        let mut reads = vec![read(vec![1, 1], -10), read(vec![-1, -1], -20)];
        select_best_diverse(&mut reads, 8);
        assert_eq!(reads.len(), 2);
    }

    #[test]
    fn an_empty_set_stays_empty() {
        let mut reads: Vec<SamplerResult> = Vec::new();
        select_best_diverse(&mut reads, 4);
        assert!(reads.is_empty());
    }
}
