//! Multi-spin coded simulated annealing.
//!
//! Section 2.7.1 of Isakov, Zintchenko, Rønnow and Troyer, *Optimized
//! simulated annealing for Ising spin glasses* (Comput. Phys. Commun. 192,
//! 265 (2015), arXiv:1401.1084). One `u64` holds the same spin across 64
//! independent replicas, so a single stream of bitwise instructions advances
//! 64 annealing runs at once.
//!
//! # Layout
//!
//! `spin[i]` bit `r` is node `i` in replica `r`, with bit `0` meaning `s = +1`
//! and bit `1` meaning `s = -1`. A replica is one read, so a 64-read job is
//! one word pass and a 36-read job is the same pass with 28 lanes idle.
//!
//! # The update
//!
//! With the protocol's energy `H = Σ h_i s_i + Σ J_uv s_u s_v`, write
//! `b_i` for the bit of `s_i` and `c_ij` for the bit of `J_ij`. Then
//! `J_ij s_i s_j = 1 - 2 l_ij` with `l_ij = c_ij ⊕ b_i ⊕ b_j`, so `l` is set
//! exactly on the replicas where the bond is satisfied, meaning it lowers
//! the energy by contributing `-1`. Summing over the
//! `d` incident bonds,
//!
//! ```text
//! m = s_i · heff_i = d - 2L,     L = Σ_j l_ij
//! ```
//!
//! and a field `h_i ∈ {-1, 0, +1}` enters as one more bond to a spin pinned at
//! `+1`, which makes `d' = d + [h_i ≠ 0]` and leaves the formula unchanged. A
//! nonzero field therefore costs one extra plane, not a separate kernel.
//!
//! `L` is accumulated as bit planes by a ripple carry-save adder, so all 64
//! replicas are counted at once. The Metropolis test in the geometric form of
//! [`crate::sa_int`] is `m ≥ -M`, which rearranges to
//!
//! ```text
//! L ≤ ⌊(d' + M) / 2⌋
//! ```
//!
//! a comparison of the bit-sliced counter against a scalar. Accepted replicas
//! come back as a mask and the flip is one XOR into `spin[i]`.
//!
//! # Cost, and what it gives up
//!
//! Nothing here is exponential in the degree: `⌈log2(d'+1)⌉ = 5` planes and a
//! carry-save tree linear in `d'`. What it gives up is the incremental
//! effective-field cache of [`crate::sampler_core`]. Each replica accepts a
//! different set of flips, so no shared cached field can exist and the local
//! field is recomputed on every attempt. At degree 20 that tax is larger than
//! it was on the degree-6 Chimera graphs of the paper.
//!
//! # Randomness
//!
//! One acceptance threshold `M` is shared by the 64 replicas of a word update,
//! which is what removes 63 of every 64 random draws. The paper takes the same
//! route and the replicas still separate, because they start from independent
//! configurations and see different `L`. It is a real coupling all the same,
//! and this crate's miner is scored on solution diversity, so
//! `mining::diversity` is measured against `cpu-sa` rather than assumed.

use quip_solver_core::CancelToken;

use crate::sa_int::{draw_row, IntGraph};
use crate::sampler_core::SampleCancelled;

/// Replicas advanced by one word update.
pub(crate) const LANES: usize = u64::BITS as usize;

/// Bit planes needed for the satisfied-bond count.
///
/// `L ≤ d' ≤ MAX_FIELD`, and [`IntGraph`] caps `MAX_FIELD` at 100, so six
/// planes (0..=63) are not enough in general — the count is clamped instead by
/// [`MAX_DEGREE`], which the builder checks.
const PLANES: usize = 6;

/// Largest `d'` this kernel accepts, set by [`PLANES`].
pub(crate) const MAX_DEGREE: usize = (1 << PLANES) - 1;

/// Spin words for one replica block.
pub(crate) struct MscState {
    /// `spin[i]` bit `r` is node `i` of replica `r`; 0 is `+1`.
    spin: Vec<u64>,
}

impl MscState {
    /// Random initial configurations, one independent draw per node covering
    /// all 64 replicas at once.
    pub(crate) fn random(n: usize, rng: &mut impl rand::Rng) -> Self {
        Self {
            spin: (0..n).map(|_| rng.gen::<u64>()).collect(),
        }
    }

    /// Replica `lane` as `±1` spins.
    pub(crate) fn lane(&self, lane: usize) -> Vec<i8> {
        self.spin
            .iter()
            .map(|w| if (w >> lane) & 1 == 0 { 1i8 } else { -1i8 })
            .collect()
    }
}

/// `d'`, the bond count including the field's ghost bond, for every node.
///
/// Returns `None` when any node exceeds [`MAX_DEGREE`] or carries a field
/// this kernel cannot fold into a single bond (`|h| > 1`).
pub(crate) fn bond_counts(graph: &IntGraph) -> Option<Vec<u8>> {
    let mut counts = Vec::with_capacity(graph.num_nodes());
    for v in 0..graph.num_nodes() {
        let h = graph.bias(v);
        if h.unsigned_abs() > 1 {
            return None;
        }
        let d = graph.neighbors(v).len() + usize::from(h != 0);
        if d > MAX_DEGREE {
            return None;
        }
        counts.push(d as u8);
    }
    Some(counts)
}

/// Advance one 64-replica block through the whole beta ladder.
///
/// Pure given `state` and `offsets`, so a test can drive this and the scalar
/// kernel of [`crate::sa_int`] from identical inputs and compare lane by lane.
pub(crate) fn anneal_word(
    graph: &IntGraph,
    counts: &[u8],
    draws: &[u8],
    sweeps_per_beta: usize,
    state: &mut MscState,
    offsets: &[usize],
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<(), SampleCancelled> {
    let n = graph.num_nodes();
    let row_len = draw_row();
    let mask = row_len - 1;
    let mut sweep = 0usize;
    for row in draws.chunks_exact(row_len) {
        for _ in 0..sweeps_per_beta {
            if let Some((guard, watermark)) = cancel {
                if guard.is_cancelled(watermark) {
                    return Err(SampleCancelled);
                }
            }
            let off = offsets.get(sweep).copied().unwrap_or(0);
            sweep += 1;
            for var in 0..n {
                let bi = state.spin[var];
                let d = usize::from(counts[var]);

                // Count satisfied bonds into bit planes. Only the planes a
                // partial count can reach are rippled, so the work is linear
                // in the degree rather than PLANES times the degree.
                let mut planes = [0u64; PLANES];
                let mut filled = 0usize;
                let h = graph.bias(var);
                if h != 0 {
                    // Ghost bond to a spin pinned at +1: l = c_h ^ b_i.
                    let l = if h < 0 { !bi } else { bi };
                    add_plane(&mut planes, &mut filled, l);
                }
                for &e in graph.neighbors(var) {
                    let sign = 0u64.wrapping_sub(u64::from(e >> 31));
                    let l = sign ^ bi ^ state.spin[(e & 0x7fff_ffff) as usize];
                    add_plane(&mut planes, &mut filled, l);
                }

                // Metropolis: accept where L <= (d + M) / 2.
                let m = usize::from(row[(var + off) & mask]);
                let limit = (d + m) / 2;
                let accept = if limit >= d {
                    u64::MAX
                } else {
                    le_constant(&planes, limit)
                };
                state.spin[var] = bi ^ accept;
            }
        }
    }
    Ok(())
}

/// Add one bit plane into the running carry-save count.
///
/// `filled` is how many inputs have been absorbed so far, which bounds how far
/// a carry can travel.
#[inline]
fn add_plane(planes: &mut [u64; PLANES], filled: &mut usize, l: u64) {
    *filled += 1;
    // A count of `filled` values needs at most this many planes.
    let height = (usize::BITS - (*filled).leading_zeros()) as usize;
    let mut carry = l;
    for p in planes.iter_mut().take(height.min(PLANES)) {
        let next = *p & carry;
        *p ^= carry;
        carry = next;
    }
}

/// Mask of lanes whose bit-sliced count is `<= limit`.
///
/// Total: a `limit` at or above [`MAX_DEGREE`] admits every representable
/// count.
///
/// Builds `[count >= limit + 1]` from the least significant plane up — with
/// `f_k` the verdict on the low `k+1` bits, `f_k = plane_k & f_{k-1}` when the
/// constant's bit is set and `f_k = plane_k | f_{k-1}` when it is clear — then
/// inverts. Written branchlessly because the constant changes on every word
/// update, so a data-dependent branch here would mispredict constantly.
#[inline]
fn le_constant(planes: &[u64; PLANES], limit: usize) -> u64 {
    let bound = limit + 1;
    if bound > MAX_DEGREE {
        // Every representable count is below the bound.
        return u64::MAX;
    }
    let mut ge = u64::MAX;
    for (k, &p) in planes.iter().enumerate() {
        let set = 0u64.wrapping_sub(((bound >> k) & 1) as u64);
        let both = p & ge;
        let either = p ^ ge;
        // set == all ones -> p & ge; set == 0 -> p | ge == (p & ge) | (p ^ ge)
        ge = both | (either & !set);
    }
    !ge
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sa_int::{anneal_from, sweep_offsets, threshold_draws};
    use quip_solver_core::beta::geometric_beta_schedule;
    use quip_solver_core::IsingGraph;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn unit_graph(n: usize, seed: u64, fields: &[f64], p: f64) -> IsingGraph {
        let mut rng = SmallRng::seed_from_u64(seed);
        let h: Vec<f64> = (0..n)
            .map(|_| fields[rng.gen_range(0..fields.len())])
            .collect();
        let mut edges = Vec::new();
        let mut j = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                if rng.gen::<f64>() < p {
                    edges.push((u, v));
                    j.push(if rng.gen::<bool>() { 1.0 } else { -1.0 });
                }
            }
        }
        IsingGraph::new(h, j, edges)
    }

    #[test]
    fn le_constant_matches_integer_comparison() {
        // Plant a known count in every lane and check the mask.
        for limit in 0..=MAX_DEGREE {
            let mut planes = [0u64; PLANES];
            for (lane, count) in (0..LANES).map(|l| (l, l % (MAX_DEGREE + 1))) {
                for (k, plane) in planes.iter_mut().enumerate() {
                    *plane |= (((count >> k) & 1) as u64) << lane;
                }
            }
            let mask = le_constant(&planes, limit);
            for lane in 0..LANES {
                let count = lane % (MAX_DEGREE + 1);
                assert_eq!(
                    (mask >> lane) & 1 == 1,
                    count <= limit,
                    "lane {lane} count {count} limit {limit}"
                );
            }
        }
    }

    #[test]
    fn add_plane_counts_set_bits() {
        let mut rng = SmallRng::seed_from_u64(4);
        for inputs in [1usize, 2, 3, 7, 20, MAX_DEGREE] {
            let words: Vec<u64> = (0..inputs).map(|_| rng.gen::<u64>()).collect();
            let mut planes = [0u64; PLANES];
            let mut filled = 0usize;
            for &w in &words {
                add_plane(&mut planes, &mut filled, w);
            }
            for lane in 0..LANES {
                let expect = words.iter().filter(|w| (*w >> lane) & 1 == 1).count();
                let got: usize = (0..PLANES)
                    .map(|k| ((planes[k] >> lane) & 1) as usize * (1 << k))
                    .sum();
                assert_eq!(got, expect, "lane {lane} with {inputs} inputs");
            }
        }
    }

    /// The load-bearing test: every lane of the multi-spin kernel must trace
    /// exactly the trajectory the scalar kernel traces from the same initial
    /// configuration and the same acceptance thresholds.
    #[test]
    fn every_lane_matches_the_scalar_kernel() {
        for (seed, fields) in [
            (1u64, &[0.0][..]),
            (2, &[-1.0, 0.0, 1.0][..]),
            (3, &[1.0][..]),
        ] {
            let graph = unit_graph(40, seed, fields, 0.35);
            let int = IntGraph::from_base(&graph).expect("unit couplings qualify");
            let counts = bond_counts(&int).expect("degree within bounds");
            let betas = geometric_beta_schedule(0.05, 6.0, 24);
            let mut rng = SmallRng::seed_from_u64(seed);
            let draws = threshold_draws(&betas, int.max_field(), &mut rng).expect("short ladder");
            let offsets = sweep_offsets(betas.len(), 2, &mut rng);

            let mut state = MscState::random(int.num_nodes(), &mut rng);
            let before: Vec<Vec<i8>> = (0..LANES).map(|l| state.lane(l)).collect();
            anneal_word(&int, &counts, &draws, 2, &mut state, &offsets, None)
                .expect("no cancel token");

            for (lane, start) in before.iter().enumerate() {
                let mut spins = start.clone();
                anneal_from(&int, &draws, 2, &mut spins, &offsets, None).expect("no cancel token");
                assert_eq!(state.lane(lane), spins, "seed {seed} lane {lane}");
            }
        }
    }

    #[test]
    fn rejects_degrees_beyond_the_plane_budget() {
        // A field larger than one cannot be a single ghost bond.
        let g = IsingGraph::new(vec![2.0, 0.0], vec![1.0], vec![(0, 1)]);
        let int = IntGraph::from_base(&g).expect("unit couplings qualify");
        assert!(bond_counts(&int).is_none());
    }
}
