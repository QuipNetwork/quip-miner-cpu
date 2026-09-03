//! Exact integer fast path for the simulated-annealing Metropolis kernel.
//!
//! Trick 1 of Isakov, Zintchenko, Rønnow and Troyer, *Optimized simulated
//! annealing for Ising spin glasses* (Comput. Phys. Commun. 192, 265 (2015),
//! arXiv:1401.1084): when the couplings are discrete with a finite range, the
//! energy change of a single flip takes only a handful of values, so the
//! Metropolis acceptance probability can be tabulated once per temperature
//! rung instead of calling `exp()` on every uphill candidate.
//!
//! # Preconditions
//!
//! - Every kept coupling is exactly `+1`, `-1`, or `0` (a zero coupling is
//!   dropped, which changes no effective field).
//! - Every field is a whole number.
//! - `max_i (|h_i| + deg_i) <= MAX_FIELD`.
//!
//! Both bundled Zephyr corpora qualify: `chain-h0` has `h = 0` and
//! `J = ±1`, `chain-ternary` has `h ∈ {-1, 0, 1}` and `J = ±1`, and the
//! maximum degree is 20. [`IntGraph::from_base`] returns `None` for anything
//! else and the caller stays on the `f64` kernel.
//!
//! # Why this is not a new sampler
//!
//! Under those preconditions the `f64` kernel's effective field is already an
//! exact integer at every step: it is seeded from a sum of `±1` and `0` terms
//! and each accepted flip adds exactly `±2`, and every such value is exact in
//! `f64`. So its Metropolis test always evaluates `exp((-delta) * beta)` with
//! `-delta = -2n` for a small integer `n`, and this module evaluates
//! `exp((-2 * beta) * n)`. `-2 * beta` is exact (a sign flip and an exponent
//! decrement), so both are a single rounding of the same real number `-2βn`
//! and produce the same `f64`. With the same random stream, both kernels
//! therefore accept and reject exactly the same flips.
//! `differential_matches_the_f64_kernel_on_unit_couplings` pins that.

use quip_solver_core::{CancelToken, IsingGraph};
use rand::rngs::SmallRng;
use rand::Rng;

use crate::sampler_core::{random_spins, SampleCancelled};

/// Largest `|h_i| + Σ_j |J_ij|` this path builds an acceptance table for.
///
/// The table costs `(MAX_FIELD + 1)` `exp()` calls per temperature rung, paid
/// once per job rather than once per uphill candidate flip, so the bound only
/// has to keep that one-off cost negligible. 100 leaves room well past the
/// degree-20 Zephyr graphs this path exists for, and keeps every reachable
/// field inside an `i8`.
const MAX_FIELD: usize = 100;

/// Coupling sign lives in the top bit of a packed neighbor entry; set means
/// `J = -1`.
const SIGN_BIT: u32 = 1 << 31;

/// Integer view of a `±1`-coupled Ising problem with whole-number fields.
///
/// Neighbor id and coupling sign share one `u32`, so a sweep streams 4 bytes
/// per incident edge instead of the `f64` kernel's 12 (a `u32` id plus an
/// `f64` coupling). On the bundled corpora that is 332 KiB of neighbor data
/// per sweep instead of 996 KiB.
pub(crate) struct IntGraph {
    h: Vec<i8>,
    /// CSR row offsets, length `n + 1`.
    nbr_start: Vec<u32>,
    /// Node id in bits 0..=30, coupling sign in bit 31.
    nbr: Vec<u32>,
    /// `max_i (|h_i| + deg_i)`, the largest `|heff|` any state can reach.
    max_field: usize,
}

impl IntGraph {
    /// Build the integer view, or `None` when `g` violates a precondition.
    ///
    /// Edges are skipped on exactly the same conditions as
    /// [`crate::sampler_core::CpuGraph`] (out-of-range endpoint, self-loop) so
    /// the two kernels see the same adjacency. A skipped edge's coupling is
    /// never inspected, so a self-loop with a non-unit coupling does not force
    /// the fallback.
    pub(crate) fn from_base(g: &IsingGraph) -> Option<Self> {
        let n = g.h.len();
        // An empty problem has no hot loop to speed up, and node ids have to
        // leave the sign bit free.
        if n == 0 || n as u64 > u64::from(!SIGN_BIT) {
            return None;
        }

        let mut h = Vec::with_capacity(n);
        for &hi in &g.h {
            if !hi.is_finite() || hi.fract() != 0.0 || hi.abs() > MAX_FIELD as f64 {
                return None;
            }
            h.push(hi as i8);
        }

        let mut deg = vec![0u32; n];
        for (k, &(u, v)) in g.edges.iter().enumerate() {
            if u >= n || v >= n || u == v {
                continue;
            }
            let coup = g.j.get(k).copied().unwrap_or(0.0);
            // A zero coupling adds nothing to either endpoint's effective
            // field, so dropping the edge leaves both kernels computing the
            // same integers and keeps this path available.
            if coup == 0.0 {
                continue;
            }
            if coup != 1.0 && coup != -1.0 {
                return None;
            }
            deg[u] += 1;
            deg[v] += 1;
        }

        let mut nbr_start = vec![0u32; n + 1];
        for i in 0..n {
            nbr_start[i + 1] = nbr_start[i] + deg[i];
        }
        let total = nbr_start[n] as usize;
        let mut nbr = vec![0u32; total];
        let mut cursor: Vec<u32> = nbr_start[..n].to_vec();
        for (k, &(u, v)) in g.edges.iter().enumerate() {
            if u >= n || v >= n || u == v {
                continue;
            }
            let coup = g.j.get(k).copied().unwrap_or(0.0);
            if coup == 0.0 {
                continue;
            }
            let sign = if coup < 0.0 { SIGN_BIT } else { 0 };
            let pu = cursor[u] as usize;
            nbr[pu] = (v as u32) | sign;
            cursor[u] += 1;
            let pv = cursor[v] as usize;
            nbr[pv] = (u as u32) | sign;
            cursor[v] += 1;
        }

        let mut max_field = 0usize;
        for i in 0..n {
            let reach = usize::from(h[i].unsigned_abs()) + deg[i] as usize;
            max_field = max_field.max(reach);
        }
        if max_field > MAX_FIELD {
            return None;
        }

        Some(Self {
            h,
            nbr_start,
            nbr,
            max_field,
        })
    }

    pub(crate) fn num_nodes(&self) -> usize {
        self.h.len()
    }

    /// Whole-number field at `var`.
    #[inline]
    pub(crate) fn bias(&self, var: usize) -> i8 {
        self.h[var]
    }

    /// Largest `|heff|` any state of this problem can reach.
    pub(crate) fn max_field(&self) -> usize {
        self.max_field
    }

    /// One row of the acceptance table, indexed by `|heff|`.
    pub(crate) fn row_len(&self) -> usize {
        self.max_field + 1
    }

    /// Packed neighbor entries for `var`.
    #[inline]
    pub(crate) fn neighbors(&self, var: usize) -> &[u32] {
        let s = self.nbr_start[var] as usize;
        let e = self.nbr_start[var + 1] as usize;
        &self.nbr[s..e]
    }
}

/// Largest acceptance table this path builds, in entries.
///
/// The table is `rungs * row_len` `f64` where the `f64` kernel needs only the
/// `rungs` of the beta ladder itself, so an unbounded rung count would cost up
/// to `MAX_FIELD + 1` times the memory `cpu-sa` used before this path existed.
/// A mining job runs at most `CPU_ADAPT.max_sweeps` rungs, so nothing in the
/// miner approaches this bound. It is here for a caller that hands the library
/// an unbounded `num_sweeps` directly: past the cap [`acceptance_table`]
/// declines, and the caller keeps the `f64` kernel, which produces the same
/// spins from the same random stream.
const MAX_TABLE_ENTRIES: usize = 1 << 20;

/// Metropolis acceptance probabilities for every temperature rung, or `None`
/// when the ladder is long enough that the table would cost more memory than
/// [`MAX_TABLE_ENTRIES`] allows.
///
/// Row `b` holds `exp(-2 * betas[b] * n)` at index `n`. Index 0 is never read:
/// a candidate with `heff` of the flipped spin's own sign is downhill or flat
/// and is accepted without consuming randomness, exactly as the `f64` kernel's
/// `delta <= 0.0` arm does.
pub(crate) fn acceptance_table(betas: &[f64], row_len: usize) -> Option<Vec<f64>> {
    if betas.len().saturating_mul(row_len) > MAX_TABLE_ENTRIES {
        return None;
    }
    let mut table = vec![0.0f64; betas.len() * row_len];
    for (b, &beta) in betas.iter().enumerate() {
        // `-2.0 * beta` is exact, so each entry is a single rounding of
        // `-2βn` and matches the argument the `f64` kernel builds as
        // `(-delta) * beta`.
        let scale = -2.0 * beta;
        for n in 1..row_len {
            table[b * row_len + n] = (scale * n as f64).exp();
        }
    }
    Some(table)
}

/// Seed the incremental effective-field cache for `spins`.
fn seed_fields(graph: &IntGraph, spins: &[i8]) -> Vec<i8> {
    let mut heff = Vec::with_capacity(graph.num_nodes());
    for v in 0..graph.num_nodes() {
        let mut f = graph.h[v];
        for &e in graph.neighbors(v) {
            let s = if spins[(e & !SIGN_BIT) as usize] > 0 {
                1
            } else {
                -1
            };
            f += if e & SIGN_BIT != 0 { -s } else { s };
        }
        heff.push(f);
    }
    heff
}

/// Flip `var` and propagate the change into its neighbors' cached fields.
#[inline]
fn flip(graph: &IntGraph, spins: &mut [i8], heff: &mut [i8], var: usize) {
    spins[var] = -spins[var];
    let ds: i8 = if spins[var] > 0 { 2 } else { -2 };
    for &e in graph.neighbors(var) {
        let node = (e & !SIGN_BIT) as usize;
        heff[node] += if e & SIGN_BIT != 0 { -ds } else { ds };
    }
}

/// Precomputed acceptance thresholds per temperature rung.
///
/// Power of two so a sweep can index it with a mask. 8192 entries is 8 KiB,
/// small enough to sit in L1 alongside the spins and the field cache.
///
/// A problem with more than `DRAW_ROW` nodes reuses a threshold within one
/// sweep, for sites `DRAW_ROW` apart. The per-sweep offset repairs the pairing
/// every sweep, so no pair of sites shares a threshold twice, but the count of
/// independent thresholds in one sweep is capped here. The bundled corpora hold
/// 4577 nodes, well inside the row.
const DRAW_ROW: usize = 8192;

// The three masked index sites below turn `& (DRAW_ROW - 1)` into `% DRAW_ROW`,
// which holds only for a power of two.
const _: () = assert!(DRAW_ROW.is_power_of_two());

/// Largest threshold table this path draws, in bytes.
///
/// One `DRAW_ROW` per rung, so the cost is `rungs * 8 KiB` however small the
/// problem. A mining job runs at most `CPU_ADAPT.max_sweeps` rungs, which costs
/// 8 MiB, so nothing in the miner approaches this bound. Past it
/// [`threshold_draws`] declines and the caller falls back to `cpu-sa`, which is
/// what every other precondition failure in these kernels does.
const MAX_DRAW_BYTES: usize = 64 << 20;

/// Largest `u64` below `p * 2^64`, saturating at `u64::MAX`.
fn scale_u64(p: f64) -> u64 {
    if p >= 1.0 {
        u64::MAX
    } else {
        // `as` saturates, and `p < 1` keeps the product inside range anyway.
        (p * 18_446_744_073_709_551_616.0) as u64
    }
}

/// Draw a table of Metropolis acceptance thresholds, one row per rung.
///
/// Trick 5 of Isakov et al.: the Metropolis test `u < exp(-β ΔE)` is the same
/// as `ΔE < -(1/β) ln u`, and the right-hand side depends only on the random
/// draw and the temperature, never on the spin. So the logarithm and the
/// exponential both leave the inner loop.
///
/// Written for integer couplings the identity is sharper still. An uphill flip
/// costs `ΔE = 2n` for an integer `n = -s_i·heff_i`, so it is accepted with
/// probability `a^n` where `a = exp(-2β)`, and drawing `M` with
/// `P(M >= n) = a^n` — a geometric variate — reproduces that exactly by
/// accepting iff `n <= M`. A downhill or flat candidate has `n <= 0 <= M`, so
/// the entire acceptance test, both directions, is the single integer
/// comparison `m >= -M`. No exponential, no floating point, no branch on the
/// sign of the energy change.
///
/// The rows are drawn once per job and shared by every read, which is what
/// removes the per-attempt random draw; the paper reuses them the same way and
/// decorrelates with a random cyclic shift, which
/// [`anneal_one_read_fast`] applies per sweep.
pub(crate) fn threshold_draws(
    betas: &[f64],
    max_field: usize,
    rng: &mut SmallRng,
) -> Option<Vec<u8>> {
    if betas.len().saturating_mul(DRAW_ROW) > MAX_DRAW_BYTES {
        return None;
    }
    let mut cut = vec![0u64; max_field + 1];
    let mut out = vec![0u8; betas.len() * DRAW_ROW];
    for (b, &beta) in betas.iter().enumerate() {
        let scale = -2.0 * beta;
        for (k, c) in cut.iter_mut().enumerate() {
            *c = scale_u64((scale * k as f64).exp());
        }
        // A problem with no bonds and no fields has no uphill move, so there
        // is no level to compare against and every threshold is 0.
        let first = cut.get(1).copied().unwrap_or(0);
        for slot in out[b * DRAW_ROW..(b + 1) * DRAW_ROW].iter_mut() {
            let u = rng.gen::<u64>();
            // `cut` is non-increasing, so the accepted levels are a prefix and
            // `M` is where that prefix ends. Over most of the ladder the
            // prefix is empty, so check that before searching.
            *slot = if u >= first {
                0
            } else {
                cut[1..].partition_point(|&c| u < c) as u8
            };
        }
    }
    Some(out)
}

/// Cyclic shifts into the threshold table, one per sweep.
///
/// Drawn up front so the annealing core is a pure function of its inputs and
/// the scalar and multi-spin kernels can be driven from the same shifts in a
/// differential test.
pub(crate) fn sweep_offsets(
    rungs: usize,
    sweeps_per_beta: usize,
    rng: &mut SmallRng,
) -> Vec<usize> {
    (0..rungs * sweeps_per_beta)
        .map(|_| (rng.gen::<u64>() as usize) & (DRAW_ROW - 1))
        .collect()
}

/// Width of one threshold row, exposed for the multi-spin kernel.
pub(crate) const fn draw_row() -> usize {
    DRAW_ROW
}

/// The tabulated-threshold annealing core: pure given `spins` and `offsets`.
pub(crate) fn anneal_from(
    graph: &IntGraph,
    draws: &[u8],
    sweeps_per_beta: usize,
    spins: &mut [i8],
    offsets: &[usize],
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<(), SampleCancelled> {
    let n = graph.num_nodes();
    let mut heff = seed_fields(graph, spins);
    const MASK: usize = DRAW_ROW - 1;

    let mut sweep = 0usize;
    for row in draws.chunks_exact(DRAW_ROW) {
        for _ in 0..sweeps_per_beta {
            if let Some((guard, watermark)) = cancel {
                if guard.is_cancelled(watermark) {
                    return Err(SampleCancelled);
                }
            }
            let off = offsets.get(sweep).copied().unwrap_or(0);
            sweep += 1;
            for var in 0..n {
                let m = if spins[var] > 0 {
                    heff[var]
                } else {
                    -heff[var]
                };
                let thr = row[(var + off) & MASK] as i8;
                if m >= -thr {
                    flip(graph, spins, &mut heff, var);
                }
            }
        }
    }
    Ok(())
}

/// One annealing read over the integer kernel.
///
/// `table` is the flat output of [`acceptance_table`] for the same beta ladder
/// the `f64` kernel would use, in rows of `graph.row_len()`. The random stream
/// is consumed in the same order and at the same points as
/// [`crate::sampler_core::anneal_one_read`].
pub(crate) fn anneal_one_read_int(
    graph: &IntGraph,
    table: &[f64],
    sweeps_per_beta: usize,
    rng: &mut SmallRng,
    cancel: Option<(&CancelToken, Option<u64>)>,
) -> Result<Vec<i8>, SampleCancelled> {
    let n = graph.num_nodes();
    let mut spins = random_spins(n, rng);

    let mut heff = seed_fields(graph, &spins);

    for row in table.chunks_exact(graph.row_len()) {
        for _ in 0..sweeps_per_beta {
            // One Relaxed load per sweep, matching the `f64` kernel's budget.
            if let Some((guard, watermark)) = cancel {
                if guard.is_cancelled(watermark) {
                    return Err(SampleCancelled);
                }
            }
            for var in 0..n {
                // `m` is the field seen along the spin's own direction, so the
                // flip costs `delta = -2m`. Downhill and flat candidates
                // (`m >= 0`) are accepted without drawing, which is what keeps
                // the random stream aligned with the `f64` kernel.
                let m = if spins[var] > 0 {
                    heff[var]
                } else {
                    -heff[var]
                };
                // Spelled with the same `u < accept_prob` the `f64` kernel
                // uses, rather than the negated `u >= accept_prob`. The two
                // disagree when the table holds a NaN, and only this order
                // rejects where the `f64` kernel rejects. `m >= 0` short
                // circuits, so an accepted downhill flip still draws nothing.
                let accept = m >= 0 || rng.gen::<f64>() < row[usize::from(m.unsigned_abs())];
                if accept {
                    flip(graph, &mut spins, &mut heff, var);
                }
            }
        }
    }
    Ok(spins)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler_core::{anneal_one_read, CpuGraph};
    use quip_solver_core::beta::geometric_beta_schedule;
    use rand::SeedableRng;

    /// A `±1`-coupled graph on `n` nodes with a random edge set and fields
    /// drawn from `fields`.
    fn unit_graph(n: usize, seed: u64, fields: &[f64]) -> IsingGraph {
        let mut rng = SmallRng::seed_from_u64(seed);
        let h: Vec<f64> = (0..n)
            .map(|_| fields[rng.gen_range(0..fields.len())])
            .collect();
        let mut edges = Vec::new();
        let mut j = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                if rng.gen::<f64>() < 0.4 {
                    edges.push((u, v));
                    j.push(if rng.gen::<bool>() { 1.0 } else { -1.0 });
                }
            }
        }
        IsingGraph::new(h, j, edges)
    }

    #[test]
    fn differential_matches_the_f64_kernel_on_unit_couplings() {
        // The whole reason `cpu-sa` may take this path without becoming a
        // different sampler: same seed, same spins, every read.
        for (seed, fields) in [
            (1u64, &[0.0][..]),
            (2, &[-1.0, 0.0, 1.0][..]),
            (3, &[-3.0, 2.0][..]),
        ] {
            let graph = unit_graph(24, seed, fields);
            let int = IntGraph::from_base(&graph).expect("unit couplings qualify");
            let cpu = CpuGraph::from_base(&graph);
            let betas = geometric_beta_schedule(0.05, 6.0, 64);
            let table = acceptance_table(&betas, int.row_len()).expect("short ladder");
            for read in 0..8u64 {
                let mut a = SmallRng::seed_from_u64(seed * 977 + read);
                let mut b = SmallRng::seed_from_u64(seed * 977 + read);
                let fast =
                    anneal_one_read_int(&int, &table, 2, &mut a, None).expect("no cancel token");
                let slow = anneal_one_read(&cpu, &betas, 2, &mut b, None).expect("no cancel token");
                assert_eq!(fast, slow, "seed {seed} read {read}");
            }
        }
    }

    #[test]
    fn rejects_problems_outside_its_preconditions() {
        // Non-unit coupling.
        let g = IsingGraph::new(vec![0.0, 0.0], vec![2.0], vec![(0, 1)]);
        assert!(IntGraph::from_base(&g).is_none());
        // Fractional field.
        let g = IsingGraph::new(vec![0.5, 0.0], vec![1.0], vec![(0, 1)]);
        assert!(IntGraph::from_base(&g).is_none());
        // Non-finite field.
        let g = IsingGraph::new(vec![f64::NAN, 0.0], vec![1.0], vec![(0, 1)]);
        assert!(IntGraph::from_base(&g).is_none());
        // Empty problem stays on the f64 path.
        let g = IsingGraph::new(vec![], vec![], vec![]);
        assert!(IntGraph::from_base(&g).is_none());
        // A missing coupling reads as 0.0 and is dropped, which leaves a
        // problem with no bonds at all — still representable.
        let g = IsingGraph::new(vec![0.0, 0.0], vec![], vec![(0, 1)]);
        let int = IntGraph::from_base(&g).expect("a dropped edge is not a violation");
        assert!(int.neighbors(0).is_empty());
    }

    #[test]
    fn skipped_edges_do_not_force_the_fallback() {
        // Self-loops and out-of-range endpoints are dropped before their
        // couplings are inspected, so a non-unit coupling on one of them is
        // not a precondition violation — it is not part of the problem.
        let g = IsingGraph::new(
            vec![0.0, 0.0],
            vec![7.0, 9.0, -1.0],
            vec![(0, 0), (0, 5), (0, 1)],
        );
        let int = IntGraph::from_base(&g).expect("only the kept edge is ±1");
        assert_eq!(int.neighbors(0), &[1 | SIGN_BIT]);
        assert_eq!(int.neighbors(1), &[SIGN_BIT]);
    }

    #[test]
    fn max_field_bounds_the_reachable_effective_field() {
        let g = unit_graph(16, 11, &[-2.0, 2.0]);
        let int = IntGraph::from_base(&g).expect("unit couplings qualify");
        let cpu = CpuGraph::from_base(&g);
        let mut rng = SmallRng::seed_from_u64(5);
        let betas = geometric_beta_schedule(0.05, 6.0, 8);
        let table = acceptance_table(&betas, int.row_len()).expect("short ladder");
        let spins = anneal_one_read_int(&int, &table, 1, &mut rng, None).expect("no cancel token");
        for v in 0..int.num_nodes() {
            let f = crate::sampler_core::effective_field(v, &spins, &cpu);
            assert!(
                f.abs() <= int.max_field as f64,
                "node {v} field {f} exceeds max_field {}",
                int.max_field
            );
        }
    }

    #[test]
    fn a_ladder_too_long_for_the_table_declines_instead_of_allocating() {
        // The caller falls back to the `f64` kernel, which produces the same
        // spins, so declining costs output nothing and bounds the memory a
        // caller can ask this path for.
        let betas = vec![1.0f64; MAX_TABLE_ENTRIES / 4 + 1];
        assert!(acceptance_table(&betas, 4).is_none());
        assert!(acceptance_table(&betas, 1).is_some());
    }

    #[test]
    fn a_ladder_too_long_for_the_threshold_row_declines_instead_of_allocating() {
        let mut rng = SmallRng::seed_from_u64(9);
        let over = vec![1.0f64; MAX_DRAW_BYTES / DRAW_ROW + 1];
        assert!(threshold_draws(&over, 4, &mut rng).is_none());
        let under = vec![1.0f64; 8];
        assert!(threshold_draws(&under, 4, &mut rng).is_some());
    }
}
