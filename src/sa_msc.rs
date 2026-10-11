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
//! a comparison of the bit-sliced counter against per-lane limits. Accepted replicas
//! come back as a mask and the flip is one XOR into `spin[i]`.
//!
//! If every lane has threshold `M` at least `d'`, the kernel inverts the spin
//! word and skips the neighbor walk.
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
//! # Randomness: `per-lane-threshold-planes-v1`
//!
//! Each lane reads its own geometric threshold `M`. A row entry holds six bit
//! planes, and bit `r` of plane `k` is bit `k` of lane `r`'s threshold. One row
//! of [`LANE_ROW`] entries is drawn per beta rung and shared by the words, and
//! each word reads it from its own seed-derived cyclic offset. Two lanes that
//! reach the same spins therefore see different thresholds, and separate
//! again (QUI-1596). The test per lane is `L ≤ ⌊(d' + M)/2⌋`, computed with a
//! bit-sliced add, a shift and a bit-sliced comparison. The row costs
//! `LANE_ROW × 64` draws per rung. This schedule differs from CUDA MSA, which
//! still shares one threshold per word.

use quip_solver_core::{CancelToken, SampleParams};
use rand::{rngs::SmallRng, Rng, SeedableRng};

use crate::coloring::Coloring;
use crate::sa_int::{draw_threshold, fill_cuts, IntGraph};
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

/// Per-lane threshold entries per beta rung. Power of two for masking.
/// Chosen by the Task 3 measurement in `docs/sa-acceleration.md`.
pub(crate) const LANE_ROW: usize = 512;
const _: () = assert!(LANE_ROW.is_power_of_two());

/// Name of the randomness scheme this kernel implements.
pub(crate) const RNG_SCHEME: &str = "per-lane-threshold-planes-v1";

/// A per-lane six-bit value, as bit planes.
type Planes = [u64; PLANES];

/// Draw one rung's per-lane thresholds. `cut` comes from
/// [`crate::sa_int::fill_cuts`]. A threshold above [`MAX_DEGREE`] accepts the
/// same moves as [`MAX_DEGREE`], so it is clamped there.
fn fill_lane_row(cut: &[u64], rng: &mut SmallRng, row: &mut [Planes]) {
    for entry in row {
        *entry = [0; PLANES];
        for lane in 0..LANES {
            let m = usize::from(draw_threshold(cut, rng.gen::<u64>())).min(MAX_DEGREE);
            for (k, plane) in entry.iter_mut().enumerate() {
                *plane |= (((m >> k) & 1) as u64) << lane;
            }
        }
    }
}

/// `⌊(d + m) / 2⌋` per lane: add the constant `d` across the planes, then
/// drop the lowest plane. `d, m ≤ 63`, so the sum needs seven planes and the
/// result six.
fn limit_planes(m: &Planes, d: usize) -> Planes {
    let mut sum = [0u64; PLANES + 1];
    let mut carry = 0u64;
    for k in 0..PLANES {
        let c = 0u64.wrapping_sub(((d >> k) & 1) as u64);
        sum[k] = m[k] ^ c ^ carry;
        carry = (m[k] & c) | (carry & (m[k] ^ c));
    }
    sum[PLANES] = carry;
    let mut out = [0u64; PLANES];
    out.copy_from_slice(&sum[1..]);
    out
}

/// Per lane `a ≤ b`, scanning from the high plane.
fn le_planes(a: &Planes, b: &Planes) -> u64 {
    let mut lt = 0u64;
    let mut eq = u64::MAX;
    for k in (0..PLANES).rev() {
        lt |= eq & !a[k] & b[k];
        eq &= !(a[k] ^ b[k]);
    }
    lt | eq
}

/// `d` in every lane, as planes.
fn constant_planes(d: usize) -> Planes {
    let mut planes = [0u64; PLANES];
    for (k, plane) in planes.iter_mut().enumerate() {
        *plane = 0u64.wrapping_sub(((d >> k) & 1) as u64);
    }
    planes
}

/// Spin words for one replica block.
pub(crate) struct MscState {
    /// `spin[i]` bit `r` is node `i` of replica `r`; 0 is `+1`.
    spin: Vec<u64>,
}

impl MscState {
    /// Random initial configurations, one independent draw per node covering
    /// all 64 replicas at once.
    pub(crate) fn random(n: usize, rng: &mut impl Rng) -> Self {
        Self {
            spin: (0..n).map(|_| rng.gen::<u64>()).collect(),
        }
    }

    /// Seeded configurations: lane `r` starts from `seeds[r]`, and each lane
    /// past the seeds keeps the draw [`MscState::random`] gives it.
    ///
    /// One `u64` is drawn per node either way, so the unseeded lanes of a
    /// partly seeded word start where the same lanes of a cold run start.
    /// That makes them a like-for-like control inside one job.
    ///
    /// `seeds` holds at most [`LANES`] states of `n` spins each; the caller
    /// checks both.
    pub(crate) fn seeded(n: usize, seeds: &[&[i8]], rng: &mut impl Rng) -> Self {
        debug_assert!(seeds.len() <= LANES);
        debug_assert!(seeds.iter().all(|s| s.len() == n));
        // Bits at and above `seeds.len()` keep the random draw.
        let keep = if seeds.len() >= LANES {
            0
        } else {
            u64::MAX << seeds.len()
        };
        let spin = (0..n)
            .map(|i| {
                let mut word = rng.gen::<u64>() & keep;
                for (lane, seed) in seeds.iter().enumerate() {
                    // Bit 1 is spin -1 in this layout.
                    if seed[i] < 0 {
                        word |= 1 << lane;
                    }
                }
                word
            })
            .collect();
        Self { spin }
    }

    /// Replica `lane` as `±1` spins.
    pub(crate) fn lane(&self, lane: usize) -> Vec<i8> {
        self.spin
            .iter()
            .map(|w| if (w >> lane) & 1 == 0 { 1i8 } else { -1i8 })
            .collect()
    }

    /// The raw per-node replica words, bit `r` of `spin[i]` being node `i` of
    /// replica `r`. Used by the bit-parallel finalisation pass, which scores
    /// all lanes of a word in one pass over the graph instead of one scalar
    /// `energy_milli` call per lane.
    pub(crate) fn spins(&self) -> &[u64] {
        &self.spin
    }

    /// Mutable view of the replica words, for a kernel that computes its own
    /// accept mask and XORs it into a node's word.
    pub(crate) fn spins_mut(&mut self) -> &mut [u64] {
        &mut self.spin
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

/// Advance all replica words together. One threshold row is redrawn per beta
/// and shared across words. Colors run in sequence, with a per-word offset.
pub(crate) fn anneal_words(
    graph: &IntGraph,
    counts: &[u8],
    colors: &Coloring,
    betas: &[f64],
    params: &SampleParams,
    states: &mut [MscState],
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<(), SampleCancelled> {
    let mut row = vec![[0u64; PLANES]; LANE_ROW];
    let mut cut = vec![0; graph.max_field() + 1];
    let mut rng = SmallRng::seed_from_u64(params.seed ^ 0x5341_5F54_424C_4531);
    let sweeps_per_beta =
        crate::sampler_core::effective_sweeps_per_beta(params.num_sweeps, params.sweeps_per_beta);
    for (beta_idx, &beta) in betas.iter().enumerate() {
        if let Some((guard, watermark)) = cancel {
            if guard.is_cancelled(watermark) {
                return Err(SampleCancelled);
            }
        }
        fill_cuts(beta, &mut cut);
        fill_lane_row(&cut, &mut rng, &mut row);
        for sweep in 0..sweeps_per_beta {
            if let Some((guard, watermark)) = cancel {
                if guard.is_cancelled(watermark) {
                    return Err(SampleCancelled);
                }
            }
            for class in colors.classes() {
                for (word, state) in states.iter_mut().enumerate() {
                    let off = word_offset(params.seed, word, beta_idx, sweep);
                    sweep_word(graph, counts, &row, class, off, state);
                }
            }
        }
    }
    Ok(())
}

/// Splitmix64 offset into the lane row for one word at one sweep.
fn word_offset(seed: u64, word: usize, beta_idx: usize, sweep: usize) -> usize {
    let mut x = (seed
        ^ (word as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
        ^ ((beta_idx as u64) << 20)
        ^ sweep as u64)
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((x ^ (x >> 31)) as usize) & (LANE_ROW - 1)
}

/// Update a color's nodes in one word from the per-lane threshold row.
fn sweep_word(
    graph: &IntGraph,
    counts: &[u8],
    row: &[Planes],
    nodes: &[u32],
    off: usize,
    state: &mut MscState,
) {
    let mask = row.len() - 1;
    for &var in nodes {
        let var = var as usize;
        let d = usize::from(counts[var]);
        let m = &row[(var + off) & mask];
        // A lane with m >= d accepts whatever its count. When every lane
        // does, skip the count.
        if le_planes(&constant_planes(d), m) == u64::MAX {
            state.spin[var] = !state.spin[var];
            continue;
        }
        let bi = state.spin[var];
        let planes = count_planes(graph, var, bi, &state.spin);
        state.spin[var] = bi ^ le_planes(&planes, &limit_planes(m, d));
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

#[cfg(test)]
mod tests {
    use super::*;
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

    fn ring(n: usize) -> IsingGraph {
        let mut rng = SmallRng::seed_from_u64(8);
        let edges = (0..n).map(|u| (u, (u + 1) % n)).collect();
        let j = (0..n)
            .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
            .collect();
        IsingGraph::new(vec![0.0; n], j, edges)
    }

    /// Plant a per-lane value `f(lane)` into six planes.
    fn planes_of(f: impl Fn(usize) -> usize) -> Planes {
        let mut planes = [0u64; PLANES];
        for lane in 0..LANES {
            let v = f(lane);
            for (k, plane) in planes.iter_mut().enumerate() {
                *plane |= (((v >> k) & 1) as u64) << lane;
            }
        }
        planes
    }

    fn lane_value(planes: &Planes, lane: usize) -> usize {
        (0..PLANES)
            .map(|k| (((planes[k] >> lane) & 1) as usize) << k)
            .sum()
    }

    #[test]
    fn le_planes_matches_integer_comparison() {
        for shift in 0..64 {
            let a = planes_of(|l| (l * 7 + shift) % 64);
            let b = planes_of(|l| (l * 13 + 3 * shift) % 64);
            let mask = le_planes(&a, &b);
            for lane in 0..LANES {
                let (x, y) = (lane_value(&a, lane), lane_value(&b, lane));
                assert_eq!((mask >> lane) & 1 == 1, x <= y, "lane {lane}: {x} <= {y}");
            }
        }
    }

    #[test]
    fn limit_planes_halves_degree_plus_threshold() {
        for d in 0..=MAX_DEGREE {
            let m = planes_of(|l| l % 64);
            let limit = limit_planes(&m, d);
            for lane in 0..LANES {
                assert_eq!(
                    lane_value(&limit, lane),
                    (d + lane % 64) / 2,
                    "d {d} lane {lane}"
                );
            }
        }
    }

    #[test]
    fn lane_rows_draw_the_geometric_threshold_per_lane() {
        let beta = 0.3;
        let mut cut = vec![0u64; 21];
        fill_cuts(beta, &mut cut);
        let mut row = vec![[0u64; PLANES]; LANE_ROW];
        fill_lane_row(&cut, &mut SmallRng::seed_from_u64(8), &mut row);
        let total = (LANE_ROW * LANES) as f64;
        for k in 1..4 {
            let at_least: usize = row
                .iter()
                .map(|e| (0..LANES).filter(|&l| lane_value(e, l) >= k).count())
                .sum();
            let want = (-2.0 * beta * k as f64).exp();
            assert!((at_least as f64 / total - want).abs() < 0.01, "k {k}");
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

    #[test]
    fn every_lane_matches_a_per_lane_scalar_reference() {
        let graph = unit_graph(40, 2, &[-1.0, 0.0, 1.0], 0.35);
        let int = IntGraph::from_base(&graph).expect("unit couplings qualify");
        let counts = bond_counts(&int).expect("degree within bounds");
        let mut cut = vec![0u64; int.max_field() + 1];
        fill_cuts(0.8, &mut cut);
        let mut row = vec![[0u64; PLANES]; LANE_ROW];
        fill_lane_row(&cut, &mut SmallRng::seed_from_u64(3), &mut row);
        let mut state = MscState::random(int.num_nodes(), &mut SmallRng::seed_from_u64(4));
        let before: Vec<Vec<i8>> = (0..LANES).map(|l| state.lane(l)).collect();
        let nodes: Vec<u32> = (0..int.num_nodes() as u32).collect();
        let off = 17;
        sweep_word(&int, &counts, &row, &nodes, off, &mut state);

        for (lane, start) in before.iter().enumerate() {
            let mut s = start.clone();
            for var in 0..int.num_nodes() {
                let d = usize::from(counts[var]);
                let m = lane_value(&row[(var + off) & (LANE_ROW - 1)], lane);
                // L: satisfied bonds, the field bond included.
                let mut l = 0;
                for &e in int.neighbors(var) {
                    let j = if e >> 31 == 1 { -1 } else { 1 };
                    let nb = (e & 0x7fff_ffff) as usize;
                    if j * i32::from(s[var]) * i32::from(s[nb]) < 0 {
                        l += 1;
                    }
                }
                let h = i32::from(int.bias(var));
                if h != 0 && h * i32::from(s[var]) < 0 {
                    l += 1;
                }
                if l <= (d + m) / 2 {
                    s[var] = -s[var];
                }
            }
            assert_eq!(state.lane(lane), s, "lane {lane}");
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
                            let row = vec![planes_of(|_| m.min(MAX_DEGREE)); LANE_ROW];
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
    fn identical_lanes_separate() {
        let graph = unit_graph(64, 6, &[0.0], 0.2);
        let int = IntGraph::from_base(&graph).unwrap();
        let counts = bond_counts(&int).unwrap();
        let colors = Coloring::new(&crate::sampler_core::CpuGraph::from_base(&graph));
        let same: Vec<i8> = (0..64).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let seeds: Vec<&[i8]> = vec![same.as_slice(); LANES];
        let mut states = vec![MscState::seeded(
            64,
            &seeds,
            &mut SmallRng::seed_from_u64(1),
        )];
        let params = SampleParams {
            seed: 9,
            sweeps_per_beta: 1,
            ..Default::default()
        };
        anneal_words(
            &int,
            &counts,
            &colors,
            &[0.2, 0.2, 0.2, 0.2],
            &params,
            &mut states,
            None,
        )
        .unwrap();
        let distinct: std::collections::BTreeSet<Vec<i8>> =
            (0..LANES).map(|l| states[0].lane(l)).collect();
        assert!(
            distinct.len() >= 32,
            "{} distinct lanes of 64",
            distinct.len()
        );
    }

    #[test]
    #[expect(clippy::print_stdout, reason = "report the measured read diversity")]
    fn dense_and_deep_runs_keep_distinct_reads() {
        let complete = unit_graph(28, 7, &[0.0], 1.0);
        let sparse_ring = ring(256);
        let mut counts = Vec::new();
        for (name, graph, sweeps) in [("complete", complete, 1000), ("ring", sparse_ring, 32_768)] {
            for count in [50, sweeps] {
                let params = SampleParams {
                    num_reads: 64,
                    num_sweeps: count,
                    seed: 12,
                    ..Default::default()
                };
                let unit = crate::research::sample_research(
                    &graph,
                    &params,
                    crate::research::ResearchKernel::UnitMsa,
                    None,
                    None,
                )
                .unwrap();
                let sa = crate::research::sample_research(
                    &graph,
                    &params,
                    crate::research::ResearchKernel::ScalarSa,
                    None,
                    None,
                )
                .unwrap();
                let unit_distinct: std::collections::BTreeSet<&Vec<i8>> =
                    unit.spins.iter().collect();
                let sa_distinct: std::collections::BTreeSet<&Vec<i8>> = sa.spins.iter().collect();
                counts.push((name, count, unit_distinct.len(), sa_distinct.len()));
            }
        }
        println!("distinct read counts (model, sweeps, unit, sa): {counts:?}");
        assert!(
            counts
                .iter()
                .all(|&(_, _, unit, sa)| unit > 1 && unit * 2 >= sa),
            "distinct read counts (model, sweeps, unit, sa): {counts:?}"
        );
    }

    #[test]
    fn models_larger_than_the_lane_row_run() {
        let graph = unit_graph(LANE_ROW * 3 + 5, 4, &[1.0], 0.0);
        let params = SampleParams {
            num_reads: 64,
            num_sweeps: 4,
            seed: 2,
            ..Default::default()
        };
        let out = crate::research::sample_research(
            &graph,
            &params,
            crate::research::ResearchKernel::UnitMsa,
            None,
            None,
        )
        .unwrap();
        assert_eq!(out.spins.len(), 64);
    }

    #[test]
    fn seeded_lanes_hold_their_seeds_and_the_rest_keep_the_random_draw() {
        let n = 50;
        let seeds: Vec<Vec<i8>> = (0..3)
            .map(|r| {
                (0..n)
                    .map(|i| if (i + r) % 3 == 0 { -1 } else { 1 })
                    .collect()
            })
            .collect();
        let views: Vec<&[i8]> = seeds.iter().map(Vec::as_slice).collect();

        let seeded = MscState::seeded(n, &views, &mut SmallRng::seed_from_u64(9));
        let random = MscState::random(n, &mut SmallRng::seed_from_u64(9));

        for (lane, seed) in seeds.iter().enumerate() {
            assert_eq!(&seeded.lane(lane), seed, "lane {lane}");
        }
        // The unseeded lanes are the control: same stream, same start.
        for lane in seeds.len()..LANES {
            assert_eq!(seeded.lane(lane), random.lane(lane), "lane {lane}");
        }
    }

    #[test]
    fn a_full_word_of_seeds_leaves_no_random_lane() {
        let n = 8;
        let seeds: Vec<Vec<i8>> = (0..LANES)
            .map(|r| {
                (0..n)
                    .map(|i| if (i ^ r) & 1 == 0 { 1 } else { -1 })
                    .collect()
            })
            .collect();
        let views: Vec<&[i8]> = seeds.iter().map(Vec::as_slice).collect();

        let seeded = MscState::seeded(n, &views, &mut SmallRng::seed_from_u64(9));

        for (lane, seed) in seeds.iter().enumerate() {
            assert_eq!(&seeded.lane(lane), seed, "lane {lane}");
        }
    }
}
