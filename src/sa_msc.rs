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
//! `L` is counted as bit planes across all 64 replicas at once. A
//! Harley-Seal carry-save tree takes at most 20 neighbors plus one field
//! bond. That tree is a port of CUDA `popcount21`. Missing inputs are
//! zero words. The field bond, if present, occupies the 21st input.
//! Wider neighborhoods use a ripple adder through the supported maximum
//! of 63 bonds including the field.
//!
//! The Metropolis test in the geometric form of [`crate::sa_int`] is
//! `m ≥ -M`, which rearranges to
//!
//! ```text
//! L ≤ ⌊(d' + M) / 2⌋
//! ```
//!
//! a comparison of the bit-sliced counter against a scalar. Accepted replicas
//! come back as a mask and the flip is one XOR into `spin[i]`.
//!
//! If the shared threshold `M` is at least `d'`, every lane accepts. The
//! kernel then inverts the spin word and skips the neighbor walk.
//!
//! # Cost, and what it gives up
//!
//! The satisfied-bond counter needs `⌈log2(d'+1)⌉` planes (five at degree 20,
//! six at the supported maximum of 63 bonds including the field).
//! What it gives up is the incremental
//! effective-field cache of [`crate::sampler_core`]. Each replica accepts a
//! different set of flips, so no shared cached field can exist and the local
//! field is recomputed on every attempt. At degree 20 that tax is larger than
//! it was on the degree-6 Chimera graphs of the paper.
//!
//! # Randomness
//!
//! One acceptance threshold `M` is shared by every replica of a node update,
//! including replicas in different words. The paper takes the same
//! route and the replicas still separate, because they start from independent
//! configurations and see different `L`. It is a real coupling all the same,
//! and this crate's miner is scored on solution diversity, so
//! `mining::diversity` is measured against `cpu-sa` rather than assumed.
//!
//! # Sweep schedule
//!
//! As in CUDA MSA, one 8192-byte threshold row is redrawn at each beta rung.
//! Each sweep uses a seed-derived cyclic offset shared by all replica words,
//! then visits independent color classes in sequence. The CPU retains its
//! `SmallRng` threshold stream, `f64` beta ladder, and per-sweep cancellation;
//! this is the same update algorithm, not a bit-identical CUDA random stream.

use quip_solver_core::{CancelToken, SampleParams};
use rand::{rngs::SmallRng, SeedableRng};

use crate::coloring::Coloring;
use crate::sa_int::{draw_row, fill_threshold_row, IntGraph};
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

/// Graph neighbors the specialized Harley-Seal counter accepts.
///
/// Matches CUDA `MSA_MAX_DEG`. The field bond, if present, occupies the
/// 21st input. Wider neighborhoods use the ripple counter.
const CSA_NEIGHBORS: usize = 20;
const CSA_INPUTS: usize = CSA_NEIGHBORS + 1;

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

/// Advance all replica words together, following the CUDA MSA schedule.
/// One threshold row is redrawn per beta and shared across words. Colors
/// run in sequence, and every word uses the same seed-derived sweep offset.
pub(crate) fn anneal_words(
    graph: &IntGraph,
    counts: &[u8],
    colors: &Coloring,
    betas: &[f64],
    params: &SampleParams,
    states: &mut [MscState],
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<(), SampleCancelled> {
    let mut row = vec![0; draw_row()];
    let mut cut = vec![0; graph.max_field() + 1];
    let mut rng = SmallRng::seed_from_u64(params.seed ^ 0x5341_5F54_424C_4531);
    for (beta_idx, &beta) in betas.iter().enumerate() {
        if let Some((guard, watermark)) = cancel {
            if guard.is_cancelled(watermark) {
                return Err(SampleCancelled);
            }
        }
        fill_threshold_row(beta, &mut cut, &mut rng, &mut row);
        for sweep in 0..params.sweeps_per_beta.max(1) {
            if let Some((guard, watermark)) = cancel {
                if guard.is_cancelled(watermark) {
                    return Err(SampleCancelled);
                }
            }
            let off = sweep_offset(params.seed, beta_idx, sweep);
            for class in colors.classes() {
                for state in states.iter_mut() {
                    sweep_word(graph, counts, &row, class, off, state);
                }
            }
        }
    }
    Ok(())
}

/// CUDA's splitmix64 offset, shared by every replica at a given sweep.
fn sweep_offset(seed: u64, beta_idx: usize, sweep: usize) -> usize {
    let mut x =
        (seed ^ ((beta_idx as u64) << 20) ^ sweep as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((x ^ (x >> 31)) as usize) & (draw_row() - 1)
}

/// Update a color's nodes in one word from a shared threshold row.
fn sweep_word(
    graph: &IntGraph,
    counts: &[u8],
    row: &[u8],
    nodes: &[u32],
    off: usize,
    state: &mut MscState,
) {
    let mask = row.len() - 1;
    for &var in nodes {
        let var = var as usize;
        let d = usize::from(counts[var]);
        let m = usize::from(row[(var + off) & mask]);
        // m >= d iff (d + m) / 2 >= d, so every lane accepts and the
        // satisfied-bond count is unused.
        if m >= d {
            state.spin[var] = !state.spin[var];
            continue;
        }
        let bi = state.spin[var];
        let planes = count_planes(graph, var, bi, &state.spin);
        let limit = (d + m) / 2;
        state.spin[var] = bi ^ le_constant(&planes, limit);
    }
}

/// Bit-sliced satisfied-bond count for every replica in `bi`.
///
/// Degree selection is outside the per-input work: at most 20 neighbors
/// plus one field bond use the specialized counter, and the rest use ripple.
fn count_planes(graph: &IntGraph, var: usize, bi: u64, spin: &[u64]) -> [u64; PLANES] {
    let nbrs = graph.neighbors(var);
    let h = graph.bias(var);
    if nbrs.len() <= CSA_NEIGHBORS {
        count_planes_csa(nbrs, h, bi, spin)
    } else {
        count_planes_ripple(nbrs, h, bi, spin)
    }
}

fn count_planes_csa(nbrs: &[u32], h: i8, bi: u64, spin: &[u64]) -> [u64; PLANES] {
    let mut x = [0u64; CSA_INPUTS];
    for (q, &e) in nbrs.iter().enumerate() {
        let sign = 0u64.wrapping_sub(u64::from(e >> 31));
        x[q] = sign ^ bi ^ spin[(e & 0x7fff_ffff) as usize];
    }
    if h != 0 {
        x[CSA_NEIGHBORS] = if h < 0 { !bi } else { bi };
    }
    popcount21(&x)
}

fn count_planes_ripple(nbrs: &[u32], h: i8, bi: u64, spin: &[u64]) -> [u64; PLANES] {
    let mut planes = [0u64; PLANES];
    let mut filled = 0usize;
    if h != 0 {
        // Ghost bond to a spin pinned at +1: l = c_h ^ b_i.
        add_plane(&mut planes, &mut filled, if h < 0 { !bi } else { bi });
    }
    for &e in nbrs {
        let sign = 0u64.wrapping_sub(u64::from(e >> 31));
        add_plane(
            &mut planes,
            &mut filled,
            sign ^ bi ^ spin[(e & 0x7fff_ffff) as usize],
        );
    }
    planes
}

/// Carry-save adder: `(carry, sum) = a + b + c` per lane.
#[inline]
fn csa(a: u64, b: u64, c: u64) -> (u64, u64) {
    let u = a ^ b;
    ((a & b) | (u & c), u ^ c)
}

/// Per-lane popcount of 21 one-bit inputs into six planes.
///
/// Missing inputs are zero words. Port of CUDA `popcount21`.
#[inline]
fn popcount21(x: &[u64; CSA_INPUTS]) -> [u64; PLANES] {
    let (t_a, ones) = csa(0, x[0], x[1]);
    let (t_b, ones) = csa(ones, x[2], x[3]);
    let (f_a, twos) = csa(0, t_a, t_b);
    let (t_a, ones) = csa(ones, x[4], x[5]);
    let (t_b, ones) = csa(ones, x[6], x[7]);
    let (f_b, twos) = csa(twos, t_a, t_b);
    let (e_a, fours) = csa(0, f_a, f_b);
    let (t_a, ones) = csa(ones, x[8], x[9]);
    let (t_b, ones) = csa(ones, x[10], x[11]);
    let (f_a, twos) = csa(twos, t_a, t_b);
    let (t_a, ones) = csa(ones, x[12], x[13]);
    let (t_b, ones) = csa(ones, x[14], x[15]);
    let (f_b, twos) = csa(twos, t_a, t_b);
    let (e_b, fours) = csa(fours, f_a, f_b);
    let (s_a, eights) = csa(0, e_a, e_b);
    let (t_a, ones) = csa(ones, x[16], x[17]);
    let (t_b, ones) = csa(ones, x[18], x[19]);
    let (f_a, twos) = csa(twos, t_a, t_b);
    let t_a = ones & x[20];
    let ones = ones ^ x[20];
    let f_b = twos & t_a;
    let twos = twos ^ t_a;
    let (e_a, fours) = csa(fours, f_a, f_b);
    let s_b = eights & e_a;
    let eights = eights ^ e_a;
    [ones, twos, fours, eights, s_a | s_b, 0]
}

/// Add one bit plane into the running count with ripple carries.
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
    use crate::sa_int::{
        anneal_from, draw_row, fill_threshold_row, sweep_offsets, threshold_draws,
    };
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

    #[test]
    fn popcount21_matches_scalar_and_ripple_on_padded_inputs() {
        let mut rng = SmallRng::seed_from_u64(21);
        for n_set in 0..=CSA_INPUTS {
            for _ in 0..8 {
                let mut x = [0u64; CSA_INPUTS];
                for slot in x.iter_mut().take(n_set) {
                    *slot = rng.gen();
                }
                let csa_planes = popcount21(&x);
                let mut ripple = [0u64; PLANES];
                let mut filled = 0usize;
                for &w in &x {
                    add_plane(&mut ripple, &mut filled, w);
                }
                for lane in 0..LANES {
                    let expect = x.iter().map(|w| ((w >> lane) & 1) as usize).sum::<usize>();
                    let from_csa: usize = (0..PLANES)
                        .map(|k| ((csa_planes[k] >> lane) & 1) as usize * (1 << k))
                        .sum();
                    let from_ripple: usize = (0..PLANES)
                        .map(|k| ((ripple[k] >> lane) & 1) as usize * (1 << k))
                        .sum();
                    assert_eq!(from_csa, expect, "csa lane {lane} n_set {n_set}");
                    assert_eq!(from_ripple, expect, "ripple lane {lane} n_set {n_set}");
                }
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
            let nodes: Vec<u32> = (0..int.num_nodes() as u32).collect();
            for (row, shifts) in draws.chunks_exact(draw_row()).zip(offsets.chunks_exact(2)) {
                for &off in shifts {
                    sweep_word(&int, &counts, row, &nodes, off, &mut state);
                }
            }

            for (lane, start) in before.iter().enumerate() {
                let mut spins = start.clone();
                anneal_from(&int, &draws, 2, &mut spins, &offsets, None).expect("no cancel token");
                assert_eq!(state.lane(lane), spins, "seed {seed} lane {lane}");
            }
        }
    }

    #[test]
    fn cancelled_generation_leaves_replica_words_untouched() {
        let graph = unit_graph(8, 1, &[0.0], 0.5);
        let int = IntGraph::from_base(&graph).expect("unit graph");
        let counts = bond_counts(&int).expect("small degree");
        let colors = Coloring::new(&crate::sampler_core::CpuGraph::from_base(&graph));
        let mut rng = SmallRng::seed_from_u64(1);
        let mut states = vec![MscState::random(8, &mut rng), MscState::random(8, &mut rng)];
        let before: Vec<_> = states.iter().map(|s| s.spin.clone()).collect();
        let token = CancelToken::default();
        token.cancel_through(7);
        let result = anneal_words(
            &int,
            &counts,
            &colors,
            &[0.1, 1.0],
            &SampleParams::default(),
            &mut states,
            Some((&token, Some(7))),
        );
        assert!(result.is_err());
        for (state, expected) in states.iter().zip(before) {
            assert_eq!(state.spin, expected);
        }
        assert!(
            anneal_words(
                &int,
                &counts,
                &colors,
                &[0.1, 1.0],
                &SampleParams::default(),
                &mut states,
                Some((&token, Some(8))),
            )
            .is_ok(),
            "a newer generation must still run"
        );
    }

    #[test]
    fn rejects_degrees_beyond_the_plane_budget() {
        // A field larger than one cannot be a single ghost bond.
        let g = IsingGraph::new(vec![2.0, 0.0], vec![1.0], vec![(0, 1)]);
        let int = IntGraph::from_base(&g).expect("unit couplings qualify");
        assert!(bond_counts(&int).is_none());
    }

    fn signed_star(neighbors: usize, field: f64, seed: u64) -> IsingGraph {
        let mut rng = SmallRng::seed_from_u64(seed);
        let n = neighbors + 1;
        let mut h = vec![0.0; n];
        h[0] = field;
        let edges: Vec<(usize, usize)> = (1..n).map(|v| (0, v)).collect();
        let j: Vec<f64> = (0..neighbors)
            .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
            .collect();
        IsingGraph::new(h, j, edges)
    }

    /// Independent per-lane popcount of satisfied bonds, including the field.
    fn scalar_satisfied(graph: &IntGraph, var: usize, spin: &[u64], lane: usize) -> u8 {
        let bi = (spin[var] >> lane) & 1;
        let mut acc = 0u8;
        let h = graph.bias(var);
        if h != 0 {
            acc += if h < 0 { 1 ^ bi } else { bi } as u8;
        }
        for &e in graph.neighbors(var) {
            let sign = u64::from(e >> 31);
            let bj = (spin[(e & 0x7fff_ffff) as usize] >> lane) & 1;
            acc += (sign ^ bi ^ bj) as u8;
        }
        acc
    }

    fn decode_planes(planes: &[u64; PLANES]) -> [u8; LANES] {
        let mut out = [0u8; LANES];
        for (lane, slot) in out.iter_mut().enumerate() {
            let mut c = 0u8;
            for (k, plane) in planes.iter().enumerate() {
                c |= (((plane >> lane) & 1) as u8) << k;
            }
            *slot = c;
        }
        out
    }

    fn threshold_samples(d: usize) -> [usize; 3] {
        let below = d.saturating_sub(1);
        [below, d, d.saturating_add(1)]
    }

    #[test]
    fn packed_counts_match_scalar_for_listed_degrees() {
        const NEIGHBORS: [usize; 7] = [0, 1, 6, 20, 21, 48, 63];
        const FIELDS: [f64; 3] = [-1.0, 0.0, 1.0];
        const SEEDS: [u64; 3] = [1, 7, 99];
        for &nbrs in &NEIGHBORS {
            for &field in &FIELDS {
                if nbrs + usize::from(field != 0.0) > MAX_DEGREE {
                    continue;
                }
                for &seed in &SEEDS {
                    let graph = signed_star(nbrs, field, seed);
                    let int = IntGraph::from_base(&graph).expect("unit star");
                    let counts = bond_counts(&int).expect("degree within bounds");
                    let n = int.num_nodes();
                    let mut rng = SmallRng::seed_from_u64(seed ^ 0x00C0_FFEE);
                    let spin_sets = [
                        MscState::random(n, &mut rng).spin,
                        vec![0u64; n],
                        vec![u64::MAX; n],
                    ];
                    for spin in &spin_sets {
                        let var = 0usize;
                        let d = usize::from(counts[var]);
                        assert_eq!(d, nbrs + usize::from(field != 0.0), "d' at center");
                        let packed = decode_planes(&count_planes(&int, var, spin[var], spin));
                        for (lane, packed_count) in packed.iter().enumerate() {
                            let expect = scalar_satisfied(&int, var, spin, lane);
                            assert_eq!(
                                *packed_count, expect,
                                "nbrs {nbrs} field {field} seed {seed} lane {lane}"
                            );
                        }
                        for m in threshold_samples(d) {
                            let mut state = MscState { spin: spin.clone() };
                            let row = vec![m as u8; draw_row()];
                            sweep_word(&int, &counts, &row, &[0], 0, &mut state);
                            let limit = (d + m) / 2;
                            for (lane, packed_count) in packed.iter().enumerate() {
                                let accept = usize::from(*packed_count) <= limit;
                                let before = (spin[var] >> lane) & 1;
                                let after = (state.spin[var] >> lane) & 1;
                                assert_eq!(
                                    after != before,
                                    accept,
                                    "nbrs {nbrs} field {field} seed {seed} m {m} lane {lane}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn early_accept_flips_every_lane_when_threshold_covers_degree() {
        let graph = signed_star(6, 1.0, 3);
        let int = IntGraph::from_base(&graph).expect("unit star");
        let counts = bond_counts(&int).expect("degree within bounds");
        let d = usize::from(counts[0]);
        let mut rng = SmallRng::seed_from_u64(5);
        let spin = MscState::random(int.num_nodes(), &mut rng).spin;
        for m in [d, d + 1] {
            let mut state = MscState { spin: spin.clone() };
            let row = vec![m as u8; draw_row()];
            sweep_word(&int, &counts, &row, &[0], 0, &mut state);
            assert_eq!(state.spin[0], !spin[0], "m {m} must flip every lane");
        }
    }

    #[test]
    fn anneal_words_matches_per_word_sweep_word() {
        let graph = unit_graph(24, 11, &[-1.0, 0.0, 1.0], 0.4);
        let int = IntGraph::from_base(&graph).expect("unit couplings qualify");
        let counts = bond_counts(&int).expect("degree within bounds");
        let colors = Coloring::new(&crate::sampler_core::CpuGraph::from_base(&graph));
        let betas = [0.2_f64, 1.5];
        let params = SampleParams {
            seed: 42,
            sweeps_per_beta: 3,
            ..Default::default()
        };
        let mut rng = SmallRng::seed_from_u64(1);
        let mut states = vec![
            MscState::random(int.num_nodes(), &mut rng),
            MscState::random(int.num_nodes(), &mut rng),
        ];
        let mut replay: Vec<MscState> = states
            .iter()
            .map(|s| MscState {
                spin: s.spin.clone(),
            })
            .collect();
        anneal_words(&int, &counts, &colors, &betas, &params, &mut states, None)
            .expect("no cancel token");

        let mut row = vec![0; draw_row()];
        let mut cut = vec![0; int.max_field() + 1];
        let mut table_rng = SmallRng::seed_from_u64(params.seed ^ 0x5341_5F54_424C_4531);
        for (beta_idx, &beta) in betas.iter().enumerate() {
            fill_threshold_row(beta, &mut cut, &mut table_rng, &mut row);
            for sweep in 0..params.sweeps_per_beta.max(1) {
                let off = sweep_offset(params.seed, beta_idx, sweep);
                for class in colors.classes() {
                    for state in replay.iter_mut() {
                        sweep_word(&int, &counts, &row, class, off, state);
                    }
                }
            }
        }
        for (got, expect) in states.iter().zip(&replay) {
            assert_eq!(got.spin, expect.spin);
        }
    }
}
