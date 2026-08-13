//! Simulated Bifurcation kernels: dSB, bSB, HdSB and HbSB over one integrator.
//!
//! The kernel evolves positions `x_i` and momenta `y_i` with symplectic Euler
//! steps. The momentum updates first, from the old positions. The position then
//! updates from the new momentum. The inelastic wall clip runs after the
//! position update. When the variant carries a heating rate, that term is added
//! last and reads the momentum captured before the step, which is the
//! integration order Kanao and Goto tuned.
//!
//! Sign convention: the SB papers minimize `-(1/2) Σ_ij J_ij s_i s_j` with both
//! index orders summed, while quip minimizes `Σ_i h_i s_i + Σ_k j_k s_u s_v`
//! with each edge listed once. Every entry of the SB matrix is therefore the
//! negated quip value: `J_uv = -j_k` and `J_{i,N} = -h_i` on the ancilla
//! column. The ancilla lives at index `N` in the state arrays only; the
//! returned spin vector has length `N`, so
//! [`quip_protocol::scoring::energy_milli`] scores the original graph unchanged.
//!
//! Parameter mapping:
//! - `num_sweeps` is the time-step count, mapped 1:1, and also sets the pump
//!   schedule `a(t_k) = (k + 1) / num_sweeps`. The step count and the schedule
//!   cannot be tuned independently: the slow ramp is the mechanism.
//! - `num_reads` is the number of independent trajectories.
//! - `seed` is the base seed; the per-read derivation matches `sampler_core`.
//! - `sweeps_per_beta` is **ignored**. SB has no temperature.
//! - `beta_range` is **ignored**, for the same reason. The control parameter is
//!   the pump `a(t)`, whose schedule is fixed by `a0 = 1` and the step count.
//!
//! Randomness enters only at initialization. After `x` and `y` are drawn the
//! dynamics are deterministic for every variant, including the heated ones, so
//! read diversity comes entirely from the initial condition.

use std::sync::atomic::{AtomicU32, Ordering};

use quip_miner_core::{IsingGraph, SampleParams, SamplerResult};
use quip_protocol::scoring::energy_milli;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::spin_barrier::SpinBarrier;

/// Internal state precision. One line changes the whole kernel to f64 for a
/// benchmark build; the reported energy never depends on it, because the
/// sampler returns `sign(x)` as `i8` and `energy_milli` rescores in f64.
type Real = f32;

/// Time step. The 2021 paper searched `{0.25, 0.5, 0.75, 1.0, 1.25}` and used
/// 1.0 or 1.25 for the ballistic and discrete variants.
const DT: Real = 1.0;

/// Detuning `a0`, set to 1 in every source. With `a0 = 1` the position update
/// reduces to `x += y * dt`.
const A0: Real = 1.0;

/// Initial draw is `x, y ~ U(-INIT_RANGE, INIT_RANGE)` per particle per read.
/// The papers say only "randomly set around zero"; 0.1 keeps particles away
/// from the walls during the early low-pump phase.
const INIT_RANGE: Real = 0.1;

/// Form of the coupling term in the SB force.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Coupling {
    /// `g(x) = sgn(x)`. Used by dSB and HdSB.
    Discrete,
    /// `g(x) = x`. Used by bSB and HbSB.
    Continuous,
}

/// One SB variant: the coupling form and the heating rate `γ`.
///
/// `γ = 0` means no heating. The nonzero values are the K2000-tuned constants
/// from Kanao and Goto 2022, fixed per variant before any tuning starts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SbVariant {
    /// Coupling form for the force term.
    pub coupling: Coupling,
    /// Heating rate. Zero disables the heating step entirely.
    pub gamma: f32,
}

/// Discrete SB. The production-track variant, shipped as `quip-cpu-sb`.
pub const DSB: SbVariant = SbVariant {
    coupling: Coupling::Discrete,
    gamma: 0.0,
};

/// Ballistic SB.
pub const BSB: SbVariant = SbVariant {
    coupling: Coupling::Continuous,
    gamma: 0.0,
};

/// Heated discrete SB.
pub const HDSB: SbVariant = SbVariant {
    coupling: Coupling::Discrete,
    gamma: 0.06,
};

/// Heated ballistic SB.
pub const HBSB: SbVariant = SbVariant {
    coupling: Coupling::Continuous,
    gamma: 0.5,
};

/// Per-particle neighbor lists in CSR layout, in `Real` precision.
///
/// Built from the base [`IsingGraph`] with the same defensive posture as
/// `sampler_core::CpuGraph`: edges out of range for `h.len()` are skipped,
/// self-loops `(u, u)` are skipped, and couplings shorter than the edge list
/// read 0.
///
/// This deliberately does not reuse `quip_miner_core::CsrGraph`, whose
/// `from_base` keeps self-loops. A self-loop in an SB neighbor row would inject
/// a spurious self-force `-c0 j_uu sgn(x_u)` into that node and shift its
/// bifurcation, while `energy_milli` scores the loop as an unoptimizable
/// constant. `sb_graph_matches_cpu_graph_adjacency` guards the duplication.
///
/// Biases live outside the CSR rows because they reach the force through the
/// ancilla column rather than through a neighbor row.
struct SbGraph {
    /// Linear biases, one per node, narrowed for the hot loop.
    h: Vec<Real>,
    /// CSR row offsets, length `n + 1`.
    nbr_start: Vec<u32>,
    /// Flattened neighbor node ids.
    nbr_node: Vec<u32>,
    /// Flattened couplings, parallel to `nbr_node`.
    nbr_coup: Vec<Real>,
    /// True when any `h_i != 0`. The ancilla exists only then.
    has_bias: bool,
    /// Coupling normalization `0.5 sqrt(n_spins - 1) / ||J||_F`, computed once
    /// per job. Zero when the problem carries no scale or the inputs are not
    /// finite.
    c0: Real,
}

impl SbGraph {
    fn from_base(g: &IsingGraph) -> Self {
        let n = g.h.len();
        let mut deg = vec![0u32; n];
        for &(u, v) in &g.edges {
            if u >= n || v >= n || u == v {
                continue;
            }
            deg[u] += 1;
            deg[v] += 1;
        }
        let mut nbr_start = vec![0u32; n + 1];
        for i in 0..n {
            nbr_start[i + 1] = nbr_start[i] + deg[i];
        }
        let total = nbr_start[n] as usize;
        let mut nbr_node = vec![0u32; total];
        let mut nbr_coup: Vec<Real> = vec![0.0; total];
        let mut cursor: Vec<u32> = nbr_start[..n].to_vec();
        // Frobenius norm accumulates in f64: tens of thousands of edges with
        // |j| near 1 lose precision in f32, and c0 divides by its square root.
        let mut sum_sq = 0.0f64;
        for (k, &(u, v)) in g.edges.iter().enumerate() {
            if u >= n || v >= n || u == v {
                continue;
            }
            let coup_f64 = g.j.get(k).copied().unwrap_or(0.0);
            sum_sq += coup_f64 * coup_f64;
            let coup = coup_f64 as Real;
            let pu = cursor[u] as usize;
            nbr_node[pu] = v as u32;
            nbr_coup[pu] = coup;
            cursor[u] += 1;
            let pv = cursor[v] as usize;
            nbr_node[pv] = u as u32;
            nbr_coup[pv] = coup;
            cursor[v] += 1;
        }
        let has_bias = g.h.iter().any(|&b| b != 0.0);
        for &b in &g.h {
            sum_sq += b * b;
        }
        // Each surviving edge contributes j^2 twice to the full symmetric
        // matrix, and each bias contributes h^2 twice (ancilla row and column).
        let norm_sq = 2.0 * sum_sq;
        let n_spins = n + usize::from(has_bias);
        let c0_f64 = if norm_sq > 0.0 && n_spins > 1 {
            0.5 * ((n_spins - 1) as f64).sqrt() / norm_sq.sqrt()
        } else {
            0.0
        };
        // Narrow after the division, then check: a tiny ||J||_F can push c0
        // past f32::MAX even when the f64 value is finite, and a non-finite h
        // or j makes the sum itself non-finite. Either way the kernel runs with
        // no coupling force and returns valid spins; the scorer decides.
        let c0_narrow = c0_f64 as Real;
        let c0 = if c0_narrow.is_finite() {
            c0_narrow
        } else {
            0.0
        };
        Self {
            h: g.h.iter().map(|&b| b as Real).collect(),
            nbr_start,
            nbr_node,
            nbr_coup,
            has_bias,
            c0,
        }
    }

    fn num_nodes(&self) -> usize {
        self.h.len()
    }

    /// Particle count: the graph's nodes plus the ancilla when the problem
    /// carries a bias.
    fn num_particles(&self) -> usize {
        self.num_nodes() + usize::from(self.has_bias)
    }

    /// `(neighbor_ids, couplings)` slices for `var`.
    #[inline]
    fn neighbors(&self, var: usize) -> (&[u32], &[Real]) {
        let s = self.nbr_start[var] as usize;
        let e = self.nbr_start[var + 1] as usize;
        (&self.nbr_node[s..e], &self.nbr_coup[s..e])
    }
}

/// Pump `a(t_k) = a0 (k + 1) / n_step`.
///
/// The `k + 1` makes the final step run at full pump, where the restoring
/// coefficient `a0 - a(t)` reaches zero and the coupling term alone decides the
/// sign of each position. Callers guarantee `n_step >= 1`.
#[inline]
fn pump(step: usize, n_step: usize) -> Real {
    A0 * (step + 1) as Real / n_step as Real
}

/// Draw one read's initial conditions, `x, y ~ U(-INIT_RANGE, INIT_RANGE)`.
///
/// Positions are drawn first, then momenta. The order is part of the kernel's
/// reproducibility contract and the determinism fixture pins it.
fn draw_initial_conditions(m: usize, rng: &mut SmallRng) -> (Vec<Real>, Vec<Real>) {
    let x: Vec<Real> = (0..m)
        .map(|_| rng.gen_range(-INIT_RANGE..=INIT_RANGE))
        .collect();
    let y: Vec<Real> = (0..m)
        .map(|_| rng.gen_range(-INIT_RANGE..=INIT_RANGE))
        .collect();
    (x, y)
}

/// Integrate `n_step` SB steps in place over the given initial conditions.
///
/// `x` and `y` have length `g.num_particles()`: the graph's nodes, plus the
/// ancilla at index `n` when the problem carries a bias.
///
/// The heating flag is a separate parameter rather than a read of
/// `variant.gamma` so a test can force the heated path on a variant whose `γ`
/// is zero. That equivalence is what makes the shipped discrete variant free of
/// the heating cost while sharing one code path with the heated variants.
fn sb_run(
    g: &SbGraph,
    n_step: usize,
    variant: SbVariant,
    heated: bool,
    x: &mut [Real],
    y: &mut [Real],
) {
    let n = g.num_nodes();
    let m = x.len();
    // `g(x_j)` for every particle, refreshed once per step so the force reads
    // the OLD positions for every i.
    let mut coupled: Vec<Real> = vec![0.0; m];
    // Momenta captured before the SB substep. Empty when the variant does not
    // heat, so the unheated path pays neither the allocation nor the copy.
    let mut y_pre: Vec<Real> = vec![0.0; if heated { m } else { 0 }];

    for k in 0..n_step {
        // `-(a0 - a(t_k))`, folded into one multiply.
        let restore = pump(k, n_step) - A0;

        match variant.coupling {
            Coupling::Discrete => {
                for (c, &xi) in coupled.iter_mut().zip(x.iter()) {
                    *c = if xi >= 0.0 { 1.0 } else { -1.0 };
                }
            }
            Coupling::Continuous => coupled.copy_from_slice(x),
        }

        if heated {
            y_pre.copy_from_slice(y);
        }

        // Momentum first, from the OLD positions. `J_uv = -j_uv` and
        // `J_{i,N} = -h_i`, so the coupling force carries a leading minus and
        // the CSR can store the quip values unchanged.
        for i in 0..n {
            let (nodes, coups) = g.neighbors(i);
            let mut f: Real = 0.0;
            for (&v, &coup) in nodes.iter().zip(coups.iter()) {
                f += coup * coupled[v as usize];
            }
            if g.has_bias {
                f += g.h[i] * coupled[n];
            }
            y[i] += (restore * x[i] - g.c0 * f) * DT;
        }
        if g.has_bias {
            let mut f: Real = 0.0;
            for (&bias, &c) in g.h.iter().zip(coupled.iter()) {
                f += bias * c;
            }
            y[n] += (restore * x[n] - g.c0 * f) * DT;
        }

        // Position from the NEW momentum, then the perfectly inelastic wall:
        // the particle stops dead at x = ±1 rather than bouncing.
        for (xi, yi) in x.iter_mut().zip(y.iter_mut()) {
            *xi += A0 * *yi * DT;
            if *xi > 1.0 {
                *xi = 1.0;
                *yi = 0.0;
            } else if *xi < -1.0 {
                *xi = -1.0;
                *yi = 0.0;
            }
        }

        // Kanao and Goto's heating: applied last, from the momentum captured
        // before this step, so a particle the wall just stopped leaves with
        // nonzero momentum. The `+γy` term is negative damping, so the
        // equations are no longer Hamiltonian and plain symplectic Euler does
        // not apply; this ordering is the one the paper tuned numerically.
        if heated {
            for (yi, &pre) in y.iter_mut().zip(y_pre.iter()) {
                *yi += variant.gamma * pre * DT;
            }
        }
    }
}

/// Integrate `n_step` SB steps across `workers` threads.
///
/// Simulated Bifurcation carries no ordering constraint inside a step: the
/// force reads the previous positions, so every particle updates independently.
/// That is the property Goto and colleagues build the method on, and it is why
/// this needs no colouring and no per-class barrier. Workers own a contiguous
/// slice of the particles for the whole run.
///
/// Two barriers per step. The first publishes `coupled` before any worker reads
/// a neighbour's entry. The second stops a fast worker starting the next step's
/// `coupled` write while a slow worker still reads the current one.
///
/// Randomness enters only at initialization, and each particle's arithmetic is
/// unchanged, so the result is bit-identical to [`sb_run`] at any worker count.
/// `parallel_matches_sequential_bit_for_bit` pins that.
fn sb_run_parallel(
    g: &SbGraph,
    n_step: usize,
    variant: SbVariant,
    heated: bool,
    x: &mut [Real],
    y: &mut [Real],
    workers: usize,
) {
    let m = x.len();
    let n = g.num_nodes();
    // f32 bit patterns behind atomics: workers write disjoint indices but read
    // every entry of `coupled`, which Rust cannot express with split borrows.
    let xa: Vec<AtomicU32> = x.iter().map(|v| AtomicU32::new(v.to_bits())).collect();
    let ya: Vec<AtomicU32> = y.iter().map(|v| AtomicU32::new(v.to_bits())).collect();
    let ca: Vec<AtomicU32> = (0..m).map(|_| AtomicU32::new(0)).collect();
    let barrier = SpinBarrier::new(workers);

    std::thread::scope(|scope| {
        for wid in 0..workers {
            let (xa, ya, ca, barrier) = (&xa, &ya, &ca, &barrier);
            scope.spawn(move || {
                let chunk = m.div_ceil(workers);
                let lo = (wid * chunk).min(m);
                let hi = (lo + chunk).min(m);
                let mut sense = false;
                let mut y_pre: Vec<Real> = vec![0.0; if heated { hi - lo } else { 0 }];

                for k in 0..n_step {
                    let restore = pump(k, n_step) - A0;

                    for i in lo..hi {
                        let xi = Real::from_bits(xa[i].load(Ordering::Relaxed));
                        let c = match variant.coupling {
                            Coupling::Discrete => {
                                if xi >= 0.0 {
                                    1.0
                                } else {
                                    -1.0
                                }
                            }
                            Coupling::Continuous => xi,
                        };
                        ca[i].store(c.to_bits(), Ordering::Relaxed);
                        if heated {
                            y_pre[i - lo] = Real::from_bits(ya[i].load(Ordering::Relaxed));
                        }
                    }
                    barrier.wait(&mut sense);

                    for i in lo..hi {
                        // The ancilla's force is a reduction over every bias, so
                        // whichever worker owns index `n` performs it. Handing
                        // it to worker 0 instead would need a third barrier.
                        let f: Real = if i == n {
                            let mut acc: Real = 0.0;
                            for (bias, c) in g.h.iter().zip(ca.iter()) {
                                acc += bias * Real::from_bits(c.load(Ordering::Relaxed));
                            }
                            acc
                        } else {
                            let (nodes, coups) = g.neighbors(i);
                            let mut acc: Real = 0.0;
                            for (&v, &coup) in nodes.iter().zip(coups.iter()) {
                                acc +=
                                    coup * Real::from_bits(ca[v as usize].load(Ordering::Relaxed));
                            }
                            if g.has_bias {
                                acc += g.h[i] * Real::from_bits(ca[n].load(Ordering::Relaxed));
                            }
                            acc
                        };

                        let xi = Real::from_bits(xa[i].load(Ordering::Relaxed));
                        let mut yi = Real::from_bits(ya[i].load(Ordering::Relaxed));
                        yi += (restore * xi - g.c0 * f) * DT;
                        let mut xi = xi + A0 * yi * DT;
                        if xi > 1.0 {
                            xi = 1.0;
                            yi = 0.0;
                        } else if xi < -1.0 {
                            xi = -1.0;
                            yi = 0.0;
                        }
                        if heated {
                            yi += variant.gamma * y_pre[i - lo] * DT;
                        }
                        xa[i].store(xi.to_bits(), Ordering::Relaxed);
                        ya[i].store(yi.to_bits(), Ordering::Relaxed);
                    }
                    barrier.wait(&mut sense);
                }
            });
        }
    });

    for (i, (xv, yv)) in x.iter_mut().zip(y.iter_mut()).enumerate() {
        *xv = Real::from_bits(xa[i].load(Ordering::Relaxed));
        *yv = Real::from_bits(ya[i].load(Ordering::Relaxed));
    }
}

/// Gauge-fix and drop the ancilla: `s_i = sgn(x_i) sgn(x_N)`.
///
/// The quadratic-only energy depends on products `s_i s_j`, so `s` and `-s`
/// have equal energy and the ancilla's own sign is the gauge. The returned
/// vector has length `num_nodes()`, matching the original graph.
///
/// A float position of exactly zero maps to `+1`, matching the reference
/// package's `positions >= 0` rule. `sampler_core::spin_sign` maps an `i8` zero
/// to `-1` instead. The two never meet: this is the only place SB crosses into
/// `i8`, and it only ever writes ±1.
fn gauge_fixed_spins(g: &SbGraph, x: &[Real]) -> Vec<i8> {
    let n = g.num_nodes();
    let gauge: i8 = if g.has_bias && x[n] < 0.0 { -1 } else { 1 };
    x.iter()
        .take(n)
        .map(|&xi| gauge * if xi >= 0.0 { 1i8 } else { -1i8 })
        .collect()
}

/// Run `num_reads` independent SB trajectories sequentially on one core.
///
/// Reads stay on a single core so the model's arrays stay hot in that core's
/// cache. Model-level parallelism (one model per core) lives in the streaming
/// pump.
///
/// # Examples
///
/// ```
/// use quip_miner_cpu::{sample_sb, IsingGraph, SampleParams, DSB};
///
/// let graph = IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)]);
/// let params = SampleParams {
///     num_reads: 4,
///     num_sweeps: 256,
///     seed: 1,
///     ..Default::default()
/// };
/// let results = sample_sb(&graph, &params, DSB);
/// assert_eq!(results.len(), params.num_reads);
/// assert!(results.iter().all(|r| r.spins.iter().all(|&s| s == 1 || s == -1)));
/// ```
pub fn sample_sb(
    graph: &IsingGraph,
    params: &SampleParams,
    variant: SbVariant,
) -> Vec<SamplerResult> {
    sample_sb_with_workers(graph, params, variant, 1)
}

/// Sample with `workers` threads splitting the particles inside every read.
///
/// Simulated Bifurcation has no ideal worker count. Throughput rises
/// monotonically with workers until the host runs out of cores, because the
/// integrator carries no ordering constraint inside a step. The output is
/// bit-identical at every count, so this setting trades latency for cores and
/// nothing else.
pub fn sample_sb_with_workers(
    graph: &IsingGraph,
    params: &SampleParams,
    variant: SbVariant,
    workers: usize,
) -> Vec<SamplerResult> {
    let num_reads = params.num_reads.max(1);
    let g = SbGraph::from_base(graph);
    let n_step = params.num_sweeps.max(1);
    let heated = variant.gamma != 0.0;
    let m = g.num_particles();
    if m == 0 {
        // No variables and no ancilla: skip the integrator entirely rather than
        // spin n_step times over empty slices.
        let energy = energy_milli(&[], &graph.h, &graph.j, &graph.edges);
        return (0..num_reads)
            .map(|_| SamplerResult {
                spins: Vec::new(),
                energy_milli: energy,
            })
            .collect();
    }
    let base_seed = params.seed;

    (0..num_reads)
        .map(|read_idx| {
            // Same derivation as `sampler_core::sample_ising`, so a benchmark
            // can pair an SB run with an SA run on one job at one seed.
            let seed = base_seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(read_idx as u64)
                .wrapping_add(1);
            let mut rng = SmallRng::seed_from_u64(seed);
            let (mut x, mut y) = draw_initial_conditions(m, &mut rng);
            if workers <= 1 {
                sb_run(&g, n_step, variant, heated, &mut x, &mut y);
            } else {
                sb_run_parallel(&g, n_step, variant, heated, &mut x, &mut y, workers);
            }
            let spins = gauge_fixed_spins(&g, &x);
            SamplerResult {
                energy_milli: energy_milli(&spins, &graph.h, &graph.j, &graph.edges),
                spins,
            }
        })
        .collect()
}

/// Test-only view of one read's final state, including the ancilla slot.
/// Mirrors `sample_sb`'s read-0 path exactly: same seed derivation, same
/// initial conditions, same heated-path selection. Plans 03 and 04 assert
/// momenta and wall positions through this window; the name and signature
/// are pinned across the qui-76 plan set.
#[cfg(test)]
pub(crate) fn sb_final_state_for_test(
    graph: &IsingGraph,
    params: &SampleParams,
    variant: SbVariant,
) -> (Vec<Real>, Vec<Real>) {
    let sb = SbGraph::from_base(graph);
    let m = sb.num_particles();
    let n_step = params.num_sweeps.max(1);
    let seed = params
        .seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(1);
    let mut rng = SmallRng::seed_from_u64(seed);
    let (mut x, mut y) = draw_initial_conditions(m, &mut rng);
    sb_run(&sb, n_step, variant, variant.gamma != 0.0, &mut x, &mut y);
    (x, y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    // Both glob imports above carry an `Rng` trait. Name rand's explicitly:
    // the glob-ambiguity lint is a future hard error.
    use rand::Rng;

    /// Hypothesis: the four shipped variants are exactly the coupling form and
    /// heating rate the design table names. A later plan builds three binaries
    /// straight off these constants, so a silent edit here would mislabel a
    /// backend on the wire.
    #[test]
    fn variant_constants_match_the_published_table() {
        assert_eq!(
            DSB,
            SbVariant {
                coupling: Coupling::Discrete,
                gamma: 0.0
            }
        );
        assert_eq!(
            BSB,
            SbVariant {
                coupling: Coupling::Continuous,
                gamma: 0.0
            }
        );
        assert_eq!(
            HDSB,
            SbVariant {
                coupling: Coupling::Discrete,
                gamma: 0.06
            }
        );
        assert_eq!(
            HBSB,
            SbVariant {
                coupling: Coupling::Continuous,
                gamma: 0.5
            }
        );
    }

    fn sb_params(num_reads: usize, num_sweeps: usize, seed: u64) -> SampleParams {
        SampleParams {
            num_reads,
            num_sweeps,
            seed,
            ..Default::default()
        }
    }

    /// Four nodes, mixed-sign couplings, nonzero biases. Exercises the ancilla,
    /// several CSR rows, and both coupling forms without being slow.
    fn mixed4() -> IsingGraph {
        IsingGraph::new(
            vec![0.5, -0.3, 0.1, 0.0],
            vec![1.0, -0.5, 0.75, -1.25, 0.25],
            vec![(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)],
        )
    }

    const ALL_VARIANTS: [SbVariant; 4] = [DSB, BSB, HDSB, HBSB];

    /// Reads whose reported energy equals `want`.
    fn count_at(results: &[SamplerResult], want: i64) -> usize {
        results.iter().filter(|r| r.energy_milli == want).count()
    }

    /// Hypothesis: the particle-parallel integrator reproduces the sequential
    /// one bit for bit at every worker count. Randomness enters only at
    /// initialization and each particle's arithmetic is unchanged, so any
    /// difference means a worker read a neighbour mid-update or the barriers
    /// let a step overlap the next one. Both are races that a tolerance-based
    /// check would hide.
    #[test]
    fn parallel_matches_sequential_bit_for_bit() {
        for g in [mixed4(), ring12(), mixed_random14()] {
            let sb = SbGraph::from_base(&g);
            let m = sb.num_particles();
            for variant in ALL_VARIANTS {
                let heated = variant.gamma != 0.0;
                let (x0, y0) = draw_initial_conditions(m, &mut SmallRng::seed_from_u64(4));

                let (mut xs, mut ys) = (x0.clone(), y0.clone());
                sb_run(&sb, 200, variant, heated, &mut xs, &mut ys);
                let bits = |v: &[Real]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();

                for workers in [2usize, 3, 4, 8] {
                    let (mut xp, mut yp) = (x0.clone(), y0.clone());
                    sb_run_parallel(&sb, 200, variant, heated, &mut xp, &mut yp, workers);
                    assert_eq!(
                        bits(&xs),
                        bits(&xp),
                        "{variant:?}: positions differ at {workers} workers"
                    );
                    assert_eq!(
                        bits(&ys),
                        bits(&yp),
                        "{variant:?}: momenta differ at {workers} workers"
                    );
                }
            }
        }
    }

    /// Hypothesis: the worker count is invisible through the public sampler
    /// too, including the ancilla path and the gauge fix.
    #[test]
    fn sample_sb_is_identical_at_every_worker_count() {
        let g = mixed_random14();
        let params = sb_params(4, 256, 31);
        let want = sample_sb_with_workers(&g, &params, DSB, 1);
        for workers in [2usize, 4, 8] {
            assert_eq!(
                sample_sb_with_workers(&g, &params, DSB, workers),
                want,
                "{workers} workers changed the result"
            );
        }
    }

    /// Hypothesis: the pump reaches `a0` exactly on the final step. Starting the
    /// index at zero instead of one would waste the last step at `a < a0`,
    /// where the restoring coefficient never reaches zero and the coupling term
    /// alone never decides the sign.
    #[test]
    fn pump_reaches_a0_exactly_on_the_final_step() {
        assert_eq!(pump(0, 1), A0);
        assert_eq!(pump(3, 4), A0);
        assert_eq!(pump(0, 4), 0.25);
        assert_eq!(pump(1, 4), 0.5);
        for k in 0..99 {
            assert!(
                pump(k, 100) < pump(k + 1, 100),
                "the pump must increase monotonically at step {k}"
            );
        }
    }

    /// Hypothesis: `J_uv = -j_uv`, so a negative quip coupling is
    /// ferromagnetic in SB and the aligned answer must be the one SB reaches
    /// more often. A sign error inverts the whole optimization silently, and it
    /// would show up here as the opposed reads outnumbering the aligned ones.
    ///
    /// The assertion counts reads rather than demanding every read be optimal.
    /// HbSB carries the largest heating rate in the family and is designed to
    /// keep fluctuating late in the run instead of freezing, so a per-read
    /// assertion would pin that variant's chaos rather than the sign
    /// convention. A strict majority is what separates a correct kernel from an
    /// inverted one, on every variant.
    #[test]
    fn ferromagnetic_pair_aligns() {
        let g = IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)]);
        for variant in ALL_VARIANTS {
            let results = sample_sb(&g, &sb_params(32, 256, 42), variant);
            assert_eq!(results.len(), 32);
            let aligned = count_at(&results, -1000);
            let opposed = count_at(&results, 1000);
            assert_eq!(
                aligned + opposed,
                32,
                "{variant:?}: this graph has only two energies"
            );
            assert!(
                aligned > opposed,
                "{variant:?}: a negative quip coupling is ferromagnetic in SB, so aligned \
                 reads must outnumber opposed ones; got {aligned} aligned, {opposed} opposed"
            );
        }
    }

    /// Hypothesis: a positive quip coupling is antiferromagnetic in SB.
    #[test]
    fn antiferromagnetic_pair_anti_aligns() {
        let g = IsingGraph::new(vec![0.0, 0.0], vec![1.0], vec![(0, 1)]);
        for variant in ALL_VARIANTS {
            let results = sample_sb(&g, &sb_params(32, 256, 43), variant);
            let opposed = count_at(&results, -1000);
            let aligned = count_at(&results, 1000);
            assert_eq!(
                aligned + opposed,
                32,
                "{variant:?}: this graph has only two energies"
            );
            assert!(
                opposed > aligned,
                "{variant:?}: a positive quip coupling is antiferromagnetic in SB, so \
                 opposed reads must outnumber aligned ones; got {opposed} opposed, \
                 {aligned} aligned"
            );
        }
    }

    /// Hypothesis: the ancilla column is `J_{i,N} = -h_i`, so a positive bias
    /// and its ancilla anti-align, and the gauge fix `s_i = sgn(x_i) sgn(x_N)`
    /// returns `-1` whichever side the ancilla itself lands on. This is the
    /// direct test of the ancilla wiring and the gauge fix together: without the
    /// gauge fix the answer would be the ancilla's coin flip.
    #[test]
    fn positive_bias_tilts_spin_negative() {
        let g = IsingGraph::new(vec![1.0], vec![], vec![]);
        for variant in ALL_VARIANTS {
            let results = sample_sb(&g, &sb_params(32, 256, 44), variant);
            let negative = count_at(&results, -1000);
            let positive = count_at(&results, 1000);
            assert_eq!(
                negative + positive,
                32,
                "{variant:?}: this graph has only two energies"
            );
            assert!(
                negative > positive,
                "{variant:?}: a positive bias must tilt the spin negative through the \
                 ancilla and the gauge fix; got {negative} negative, {positive} positive"
            );
        }
    }

    /// Hypothesis: the bias enters the normalization as well as the force.
    /// Dropping it from `||J||_F` would leave `c0` too large on a bias-heavy
    /// instance and let the bias overwhelm the coupling, which the first three
    /// sign tests cannot detect.
    ///
    /// `h = [2.0, -0.5]` with one ferromagnetic edge `j = -1.0`. Per-spin greedy
    /// on the biases alone picks `(-1, +1)` at -1500. The optimum `(-1, -1)` at
    /// -2500 forces node 1 against its own bias, because the coupling is
    /// stronger than that bias.
    #[test]
    fn mixed_bias_and_coupling_beats_greedy() {
        let g = IsingGraph::new(vec![2.0, -0.5], vec![-1.0], vec![(0, 1)]);
        for variant in ALL_VARIANTS {
            let results = sample_sb(&g, &sb_params(32, 512, 45), variant);
            let best = results
                .iter()
                .map(|r| r.energy_milli)
                .min()
                .expect("num_reads > 0");
            assert_eq!(
                best,
                -2500,
                "{variant:?}: SB must reach the coupling-driven optimum (-1, -1), got {:?}",
                results
                    .iter()
                    .map(|r| (r.spins.clone(), r.energy_milli))
                    .collect::<Vec<_>>()
            );
            assert!(
                results
                    .iter()
                    .any(|r| r.spins == vec![-1i8, -1i8] && r.energy_milli == -2500),
                "{variant:?}: the optimal energy must come from the optimal spins"
            );
        }
    }

    /// Hypothesis: every reported energy equals consensus scoring of its own
    /// spins, and every spin is a valid wire value.
    #[test]
    fn reported_energies_match_consensus_scoring() {
        let g = mixed4();
        for variant in ALL_VARIANTS {
            for r in sample_sb(&g, &sb_params(8, 256, 46), variant) {
                assert!(r.spins.iter().all(|&s| s == 1 || s == -1));
                assert_eq!(r.spins.len(), g.h.len());
                assert_eq!(
                    r.energy_milli,
                    energy_milli(&r.spins, &g.h, &g.j, &g.edges),
                    "{variant:?}: reported energy must equal consensus scoring"
                );
            }
        }
    }

    /// Hypothesis: randomness enters only at initialization, so the same seed
    /// and the same parameters give byte-identical results for every variant,
    /// including the heated ones. The protocol requires that the same seed and
    /// the same binary produce the same output.
    #[test]
    fn same_seed_produces_identical_results() {
        let g = mixed_random14();
        let params = sb_params(8, 256, 777);
        for variant in ALL_VARIANTS {
            let a = sample_sb(&g, &params, variant);
            let b = sample_sb(&g, &params, variant);
            assert_eq!(a, b, "{variant:?} must be deterministic at a fixed seed");
        }
    }

    /// Hypothesis: reads are independent restarts and the base seed changes all
    /// of them, so read diversity comes entirely from the initial condition.
    /// A kernel that ignored the seed would still pass every other test here.
    ///
    /// The step count is 32 rather than the 256 the other tests use. At 14
    /// nodes dSB reaches the exact ground state from every initial condition
    /// well before 256 steps, so both seeds return the same optimum on every
    /// read and the comparison stops discriminating. Thirty-two steps keeps the
    /// trajectories short of convergence, which is where the seed is still
    /// visible in the output.
    #[test]
    fn different_seeds_produce_different_results() {
        let g = mixed_random14();
        let a = sample_sb(&g, &sb_params(8, 32, 1), DSB);
        let b = sample_sb(&g, &sb_params(8, 32, 2), DSB);
        assert_ne!(
            a, b,
            "randomized initial conditions must diversify across seeds"
        );
    }

    /// Regression guard on DT, INIT_RANGE and c0.
    ///
    /// Particles reaching the walls early is intrinsic to SB: c0 is chosen so
    /// the coupling force and the restoring force are the same order when the
    /// pump starts. The pathology is different. With the constants too large
    /// every particle pins in the opening steps and never moves again, and the
    /// run returns nothing but the sign of its initial draw. Guard that
    /// directly, and check the wall invariant on the same run.
    ///
    /// This is a regression guard on the constants, not a claim about the
    /// algorithm.
    #[test]
    fn final_positions_are_not_just_the_sign_of_the_initial_draw() {
        let g = mixed_random14();
        let sb = SbGraph::from_base(&g);
        let m = sb.num_particles();
        let (x0, y0) = draw_initial_conditions(m, &mut SmallRng::seed_from_u64(31));
        let (mut x, mut y) = (x0.clone(), y0);
        sb_run(&sb, 1000, DSB, false, &mut x, &mut y);

        let moved = x0
            .iter()
            .zip(x.iter())
            .filter(|(a, b)| (**a >= 0.0) != (**b >= 0.0))
            .count();
        assert!(
            moved * 4 >= m,
            "only {moved} of {m} particles left the side they started on: the \
             constants pin the system before the pump separates anything"
        );
        assert!(
            x.iter().all(|v| v.abs() <= 1.0),
            "the wall clip must hold |x| <= 1: {x:?}"
        );
        assert!(
            x.iter().all(|v| v.is_finite()) && y.iter().all(|v| v.is_finite()),
            "the integrator must not diverge: x = {x:?}, y = {y:?}"
        );
    }

    /// Minimum energy over all `2^n` configurations, in milli units. Only safe
    /// for `n <= 24`; every caller here stays at or below 14.
    fn brute_force_min_energy(g: &IsingGraph) -> i64 {
        let n = g.h.len();
        let mut best = i64::MAX;
        for mask in 0u32..(1u32 << n) {
            let spins: Vec<i8> = (0..n)
                .map(|i| if (mask >> i) & 1 == 1 { 1i8 } else { -1i8 })
                .collect();
            best = best.min(energy_milli(&spins, &g.h, &g.j, &g.edges));
        }
        best
    }

    /// Every edge antiferromagnetic, so no assignment satisfies all three.
    /// Six of the eight configurations are ground states.
    fn frustrated_triangle() -> IsingGraph {
        IsingGraph::new(
            vec![0.0; 3],
            vec![1.0, 1.0, 1.0],
            vec![(0, 1), (1, 2), (0, 2)],
        )
    }

    /// Twelve-node ring with alternating coupling signs and one nonzero bias.
    fn ring12() -> IsingGraph {
        let edges: Vec<(usize, usize)> = (0..12).map(|i| (i, (i + 1) % 12)).collect();
        let j: Vec<f64> = (0..12)
            .map(|k| if k % 2 == 0 { -1.0 } else { 1.0 })
            .collect();
        let mut h = vec![0.0; 12];
        h[0] = 0.5;
        IsingGraph::new(h, j, edges)
    }

    /// Fourteen nodes, mixed-sign couplings and nonzero biases everywhere,
    /// drawn once from a fixed seed so the fixture is stable.
    fn mixed_random14() -> IsingGraph {
        let mut rng = SmallRng::seed_from_u64(20_260_807);
        let h: Vec<f64> = (0..14).map(|_| rng.gen_range(-1.0..=1.0)).collect();
        let mut edges = Vec::new();
        for u in 0..14 {
            for v in (u + 1)..14 {
                if rng.gen::<f64>() < 0.35 {
                    edges.push((u, v));
                }
            }
        }
        let j: Vec<f64> = (0..edges.len())
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();
        IsingGraph::new(h, j, edges)
    }

    /// Hypothesis: every variant reaches the true ground state of a small
    /// frustrated instance. The triangle is tiny and six of its eight
    /// configurations are optimal, so this is the cheap correctness gate that
    /// exercises all four variant switches.
    ///
    /// Do not raise the sweep count here. The triangle is the one fixture whose
    /// particles all share an identical neighborhood and carry no bias, so the
    /// diagonal `x0 = x1 = x2` is invariant under the dynamics. The frustrated
    /// off-diagonal directions have no stable fixed point, so under a discrete
    /// coupling the diagonal is the only attracting direction the pump leaves,
    /// and it corresponds to the all-aligned state: the *worst* state of an
    /// antiferromagnet. Longer integration converges onto it. Measured for DSB
    /// at 128 reads, the ground-state hit rate falls 105/128 at 512 sweeps to
    /// 32/128 at 2048 and 0/128 at 8192. The ballistic variants hold 128/128
    /// throughout, because a continuous coupling preserves the magnitude
    /// differences that break the symmetry. No other fixture shows this: ring12,
    /// mixed_random14, and a random 16-node frustrated instance all stay at
    /// 64/64 for every variant out to 8192 sweeps.
    #[test]
    fn every_variant_finds_the_frustrated_triangle_ground_state() {
        let g = frustrated_triangle();
        let want = brute_force_min_energy(&g);
        assert_eq!(want, -1000, "triangle ground energy");
        for variant in ALL_VARIANTS {
            let results = sample_sb(&g, &sb_params(32, 512, 2026), variant);
            let best = results
                .iter()
                .map(|r| r.energy_milli)
                .min()
                .expect("num_reads > 0");
            assert_eq!(best, want, "{variant:?}: best {best} != brute-force {want}");
        }
    }

    /// Hypothesis: every variant reaches the true ground state of larger
    /// instances at a sensible step count.
    ///
    /// This originally covered the discrete variants alone, because Kanao and
    /// Goto rank solution quality HbSB > HdSB > dSB > bSB and reaching an exact
    /// ground state was not a claim to make for bSB on faith. Measured on these
    /// fixtures, both ballistic variants reach every ground state here, so the
    /// exclusion was caution rather than a property of the kernel. That ranking
    /// is about hard instances at scale, which these fixtures are not.
    ///
    /// A failure is a finding about the kernel or the constants; report it
    /// rather than weakening the assertion or raising the budget without
    /// recording why.
    #[test]
    fn every_variant_finds_brute_forced_ground_states() {
        let cases: Vec<(&str, IsingGraph, usize, usize)> = vec![
            ("ring12", ring12(), 64, 2048),
            ("mixed_random14", mixed_random14(), 128, 4096),
        ];
        for (name, g, reads, sweeps) in cases {
            let want = brute_force_min_energy(&g);
            for variant in ALL_VARIANTS {
                let results = sample_sb(&g, &sb_params(reads, sweeps, 2026), variant);
                let best = results
                    .iter()
                    .map(|r| r.energy_milli)
                    .min()
                    .expect("num_reads > 0");
                assert_eq!(
                    best, want,
                    "{name} / {variant:?}: best {best} != brute-force ground {want}"
                );
            }
        }
    }

    /// Hypothesis: a graph with no variables returns `num_reads` empty spin
    /// vectors at energy 0 without allocating state or running the integrator.
    #[test]
    fn empty_graph_returns_empty_reads() {
        let g = IsingGraph::new(vec![], vec![], vec![]);
        for variant in ALL_VARIANTS {
            let results = sample_sb(&g, &sb_params(3, 8192, 1), variant);
            assert_eq!(results.len(), 3);
            assert!(results
                .iter()
                .all(|r| r.spins.is_empty() && r.energy_milli == 0));
        }
    }

    /// Hypothesis: the remaining rows of the design's edge-case table all return
    /// valid ±1 spins of the right length and never panic. `clippy::panic` is
    /// denied and a malformed job must not take down a mining session.
    #[test]
    fn degenerate_graphs_return_valid_spins() {
        let cases: Vec<(&str, IsingGraph)> = vec![
            (
                "single node, no bias",
                IsingGraph::new(vec![0.0], vec![], vec![]),
            ),
            (
                "single node, bias only",
                IsingGraph::new(vec![1.0], vec![], vec![]),
            ),
            (
                "all zero",
                IsingGraph::new(vec![0.0, 0.0, 0.0], vec![0.0, 0.0], vec![(0, 1), (1, 2)]),
            ),
            (
                "disconnected node",
                IsingGraph::new(vec![0.0, 0.0, 0.0], vec![-1.0], vec![(0, 1)]),
            ),
            (
                "non-finite bias",
                IsingGraph::new(vec![f64::NAN, 1.0], vec![-1.0], vec![(0, 1)]),
            ),
            (
                "non-finite coupling",
                IsingGraph::new(vec![0.0, 0.0], vec![f64::INFINITY], vec![(0, 1)]),
            ),
            (
                "short j vector",
                IsingGraph::new(vec![0.5, 0.0], vec![], vec![(0, 1)]),
            ),
        ];
        for (name, g) in cases {
            for variant in ALL_VARIANTS {
                let results = sample_sb(&g, &sb_params(4, 128, 3), variant);
                assert_eq!(results.len(), 4, "{name} / {variant:?}");
                for r in &results {
                    assert_eq!(r.spins.len(), g.h.len(), "{name} / {variant:?}");
                    assert!(
                        r.spins.iter().all(|&s| s == 1 || s == -1),
                        "{name} / {variant:?}: spins must be ±1, got {:?}",
                        r.spins
                    );
                }
            }
        }
    }

    /// Hypothesis (Kanao and Goto): the heating increment is applied after the
    /// wall clip and reads the momentum from before the step, so a particle the
    /// wall just stopped leaves the step moving. Plain SB leaves it at exactly
    /// zero. That difference is the whole mechanism by which the heated
    /// variants re-escape local optima that plain SB freezes into.
    ///
    /// Both particles start just short of the +1 wall with outward momentum, so
    /// step 0 clips both.
    #[test]
    fn heating_restores_momentum_after_a_wall_collision() {
        let g = IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)]);
        let sb = SbGraph::from_base(&g);

        let mut x_cold: Vec<Real> = vec![0.99, 0.99];
        let mut y_cold: Vec<Real> = vec![0.5, 0.5];
        sb_run(&sb, 1, DSB, false, &mut x_cold, &mut y_cold);
        assert_eq!(x_cold, vec![1.0, 1.0], "the wall must clip both positions");
        assert_eq!(
            y_cold,
            vec![0.0, 0.0],
            "plain SB leaves a wall-stopped particle at rest"
        );

        let mut x_hot: Vec<Real> = vec![0.99, 0.99];
        let mut y_hot: Vec<Real> = vec![0.5, 0.5];
        sb_run(&sb, 1, HDSB, true, &mut x_hot, &mut y_hot);
        assert_eq!(x_hot, vec![1.0, 1.0], "the wall must clip both positions");
        for &yi in &y_hot {
            assert_ne!(
                yi, 0.0,
                "heating must leave nonzero momentum at the wall, got {yi}"
            );
        }
    }

    /// Hypothesis: with `γ = 0` the heating increment is exactly zero, so
    /// forcing the heated path on an unheated variant must not move a single
    /// bit. This is what lets the shipped discrete binary skip the heating work
    /// without becoming a second implementation.
    #[test]
    fn gamma_zero_through_the_heated_path_is_byte_identical() {
        let g = mixed4();
        let sb = SbGraph::from_base(&g);
        let m = sb.num_particles();
        let (x0, y0) = draw_initial_conditions(m, &mut SmallRng::seed_from_u64(9));

        let (mut x_cold, mut y_cold) = (x0.clone(), y0.clone());
        sb_run(&sb, 200, DSB, false, &mut x_cold, &mut y_cold);
        let (mut x_hot, mut y_hot) = (x0, y0);
        sb_run(&sb, 200, DSB, true, &mut x_hot, &mut y_hot);

        let bits = |v: &[Real]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&x_cold),
            bits(&x_hot),
            "positions must be byte-identical with gamma = 0"
        );
        assert_eq!(
            bits(&y_cold),
            bits(&y_hot),
            "momenta must be byte-identical with gamma = 0"
        );
    }

    /// Hypothesis: the test window mirrors `sample_sb`'s read-0 path. The
    /// state carries one extra slot when the graph has a bias (the ancilla),
    /// walls hold, and gauge-fixing the window's positions reproduces
    /// `sample_sb`'s read-0 spins for the same seed.
    #[test]
    fn final_state_window_matches_read_zero() {
        let g = IsingGraph::new(vec![0.5, -0.25], vec![-1.0], vec![(0, 1)]);
        let params = SampleParams {
            num_reads: 1,
            num_sweeps: 64,
            seed: 3,
            ..Default::default()
        };
        let (x, y) = sb_final_state_for_test(&g, &params, DSB);
        assert_eq!(x.len(), 3, "two nodes plus the ancilla");
        assert_eq!(y.len(), 3);
        assert!(x.iter().all(|v| v.abs() <= 1.0), "walls must hold");

        let sb = SbGraph::from_base(&g);
        let spins = gauge_fixed_spins(&sb, &x);
        let results = sample_sb(&g, &params, DSB);
        assert_eq!(
            results[0].spins, spins,
            "the window must reproduce read 0 exactly"
        );
    }

    /// Hypothesis: only the two heated variants select the heating path.
    #[test]
    fn only_heated_variants_carry_a_nonzero_gamma() {
        assert_eq!(DSB.gamma, 0.0);
        assert_eq!(BSB.gamma, 0.0);
        const { assert!(HDSB.gamma > 0.0) };
        const { assert!(HBSB.gamma > 0.0) };
    }

    /// Self-loop, out-of-range edge, and a `j` vector shorter than `edges`. All
    /// three defensive cases in one graph.
    fn adversarial() -> IsingGraph {
        IsingGraph::new(
            vec![0.5, -0.25, 0.0],
            vec![2.0, -1.0, 0.75],
            vec![(0, 0), (0, 1), (9, 2), (1, 2)],
        )
    }

    /// Hypothesis: `SbGraph` repeats `CpuGraph`'s CSR construction rather than
    /// reusing `quip_miner_core::CsrGraph`, whose `from_base` keeps self-loops.
    /// A self-loop in an SB neighbor row injects a spurious self-force
    /// `-c0 j_uu sgn(x_u)` that biases that node's bifurcation, while
    /// `energy_milli` scores the loop as an unoptimizable constant. This test
    /// replaces hoisting a shared builder: it pins the duplicate against the
    /// original.
    ///
    /// The fixture's couplings are dyadic, so narrowing them to f32 and widening
    /// back to f64 is exact and the comparison is meaningful.
    #[test]
    fn sb_graph_matches_cpu_graph_adjacency() {
        let g = adversarial();
        let sb = SbGraph::from_base(&g);
        let cpu = crate::sampler_core::CpuGraph::from_base(&g);
        assert_eq!(sb.num_nodes(), cpu.num_nodes());
        for v in 0..sb.num_nodes() {
            let (sb_nodes, sb_coups) = sb.neighbors(v);
            let (cpu_nodes, cpu_coups) = cpu.neighbors(v);
            assert_eq!(sb_nodes, cpu_nodes, "neighbor ids differ at node {v}");
            let widened: Vec<f64> = sb_coups.iter().map(|&c| f64::from(c)).collect();
            assert_eq!(widened, cpu_coups, "couplings differ at node {v}");
        }
    }

    /// Hypothesis: the CSR drops a self-loop and an out-of-range edge outright,
    /// and a missing coupling reads 0.0 rather than shifting the edge list.
    #[test]
    fn sb_graph_drops_self_loops_and_out_of_range_edges() {
        let sb = SbGraph::from_base(&adversarial());
        assert_eq!(sb.neighbors(0).0, &[1]);
        assert_eq!(sb.neighbors(0).1, &[-1.0]);
        assert_eq!(sb.neighbors(1).0, &[0, 2]);
        assert_eq!(sb.neighbors(1).1, &[-1.0, 0.0]);
        assert_eq!(sb.neighbors(2).0, &[1]);
        assert_eq!(sb.neighbors(2).1, &[0.0]);
        assert!(sb.has_bias, "h = [0.5, -0.25, 0.0] carries a bias");
    }

    /// Hypothesis: `c0 = 0.5 sqrt(n_spins - 1) / ||J||_F` with
    /// `||J||_F^2 = 2 (Σ j^2 + Σ h^2)` and `n_spins = N + 1` when the ancilla
    /// exists. Here N = 2, one edge with j = -1, biases [1.0, 0.0], so
    /// `||J||_F^2 = 2(1 + 1) = 4` and `n_spins = 3`.
    #[test]
    fn c0_matches_the_closed_form_with_the_ancilla() {
        let g = IsingGraph::new(vec![1.0, 0.0], vec![-1.0], vec![(0, 1)]);
        let sb = SbGraph::from_base(&g);
        let want = 0.5 * 2.0f64.sqrt() / 4.0f64.sqrt();
        assert!(
            (f64::from(sb.c0) - want).abs() < 1e-6,
            "c0 = {} want {want}",
            sb.c0
        );
    }

    /// Hypothesis: only edges that survive the defensive filter may enter
    /// `Σ j^2`. Counting a self-loop or an out-of-range edge in the
    /// normalization while the force loop skips it would shrink `c0` on any
    /// graph that carries one, and the two would disagree silently.
    #[test]
    fn self_loop_and_out_of_range_edges_are_excluded_from_c0() {
        let plain = IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)]);
        let looped = IsingGraph::new(vec![0.0, 0.0], vec![-1.0, 50.0], vec![(0, 1), (1, 1)]);
        let ranged = IsingGraph::new(vec![0.0, 0.0], vec![-1.0, 50.0], vec![(0, 1), (0, 9)]);
        let want = SbGraph::from_base(&plain).c0;
        assert_eq!(SbGraph::from_base(&looped).c0, want);
        assert_eq!(SbGraph::from_base(&ranged).c0, want);
    }

    /// Hypothesis: the zero guard covers every problem with no scale to
    /// normalize against. `c0` is the only division in the kernel, so it must
    /// never run on a zero denominator.
    #[test]
    fn c0_is_zero_when_the_problem_carries_no_scale() {
        let empty = SbGraph::from_base(&IsingGraph::new(vec![], vec![], vec![]));
        assert_eq!(empty.c0, 0.0);
        let all_zero =
            SbGraph::from_base(&IsingGraph::new(vec![0.0, 0.0], vec![0.0], vec![(0, 1)]));
        assert_eq!(all_zero.c0, 0.0);
        let single = SbGraph::from_base(&IsingGraph::new(vec![0.0], vec![], vec![]));
        assert_eq!(single.c0, 0.0);
    }

    /// Hypothesis: a non-finite bias or coupling on the wire must not produce a
    /// non-finite `c0`. `energy_milli` already scores a non-finite problem as
    /// its `1 << 62` sentinel, so the kernel matches that posture instead of
    /// adding validation. It must not panic: `clippy::panic` is denied and a
    /// malformed job must not take down a mining session.
    #[test]
    fn c0_is_zero_for_non_finite_biases_or_couplings() {
        let nan_bias = SbGraph::from_base(&IsingGraph::new(
            vec![f64::NAN, 0.0],
            vec![-1.0],
            vec![(0, 1)],
        ));
        assert_eq!(nan_bias.c0, 0.0);
        let inf_coupling = SbGraph::from_base(&IsingGraph::new(
            vec![0.0, 0.0],
            vec![f64::INFINITY],
            vec![(0, 1)],
        ));
        assert_eq!(inf_coupling.c0, 0.0);
    }

    /// Hypothesis: the bias belongs in the normalization. The ancilla
    /// construction forces it, because bias values become matrix entries that
    /// SB normalizes. A bias-only problem is solvable and must get a positive
    /// `c0` through the ancilla column.
    #[test]
    fn bias_only_graph_gets_a_positive_c0_through_the_ancilla() {
        let sb = SbGraph::from_base(&IsingGraph::new(vec![1.0], vec![], vec![]));
        assert!(sb.has_bias);
        assert!(
            sb.c0 > 0.0,
            "bias-only graph must normalize through the ancilla, got {}",
            sb.c0
        );
    }

    /// Hypothesis: an all-zero bias vector means no ancilla.
    #[test]
    fn zero_biases_mean_no_ancilla() {
        let g = IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)]);
        assert!(!SbGraph::from_base(&g).has_bias);
    }

    /// Hypothesis: the integrator constants are the paper values. The
    /// determinism fixture and the stuck-at-wall guard both depend on them.
    #[test]
    fn integrator_constants_are_the_paper_values() {
        assert_eq!(DT, 1.0);
        assert_eq!(A0, 1.0);
        assert_eq!(INIT_RANGE, 0.1);
    }

    /// Build a small random graph from proptest inputs. Edge endpoints are taken
    /// modulo `n`, which deliberately generates self-loops and repeated edges.
    fn graph_from_parts(n: usize, h: Vec<f64>, edge_data: Vec<(u8, u8, f64)>) -> IsingGraph {
        let n = n.min(h.len()).max(1);
        let h: Vec<f64> = h.into_iter().take(n).collect();
        let mut edges = Vec::with_capacity(edge_data.len());
        let mut j = Vec::with_capacity(edge_data.len());
        for (u, v, c) in edge_data {
            edges.push(((u as usize) % n, (v as usize) % n));
            j.push(c);
        }
        IsingGraph::new(h, j, edges)
    }

    proptest! {
        // No file persistence: keeps the worktree free of proptest-regressions/.
        #![proptest_config(ProptestConfig {
            cases: 64,
            failure_persistence: None,
            ..ProptestConfig::default()
        })]

        /// Hypothesis: every returned spin is a valid wire value, the vector
        /// length always matches the graph, and the reported energy always
        /// equals consensus scoring of those exact spins.
        #[test]
        fn prop_spins_are_pm1_and_energy_matches_scoring(
            n in 1usize..=6,
            h in prop::collection::vec(-2.0f64..2.0, 1..=6),
            edge_data in prop::collection::vec(
                (any::<u8>(), any::<u8>(), -2.0f64..2.0),
                0..=12,
            ),
            seed in any::<u64>(),
            sweeps in 1usize..=64,
        ) {
            let g = graph_from_parts(n, h, edge_data);
            for variant in ALL_VARIANTS {
                let results = sample_sb(&g, &sb_params(3, sweeps, seed), variant);
                prop_assert_eq!(results.len(), 3);
                for r in &results {
                    prop_assert_eq!(r.spins.len(), g.h.len());
                    prop_assert!(r.spins.iter().all(|&s| s == 1 || s == -1));
                    prop_assert_eq!(
                        r.energy_milli,
                        energy_milli(&r.spins, &g.h, &g.j, &g.edges)
                    );
                }
            }
        }

        /// Hypothesis: the integrator never produces a non-finite value and the
        /// wall rule holds after every step. This catches an integrator that
        /// diverges at a large time step and a wall rule applied in the wrong
        /// order. HbSB is used because its heating rate is the largest, so it
        /// is the variant most able to run away.
        #[test]
        fn prop_no_nan_and_walls_hold(
            n in 1usize..=6,
            h in prop::collection::vec(-2.0f64..2.0, 1..=6),
            edge_data in prop::collection::vec(
                (any::<u8>(), any::<u8>(), -2.0f64..2.0),
                0..=12,
            ),
            seed in any::<u64>(),
            sweeps in 1usize..=64,
        ) {
            let g = graph_from_parts(n, h, edge_data);
            let sb = SbGraph::from_base(&g);
            let m = sb.num_particles();
            let (mut x, mut y) =
                draw_initial_conditions(m, &mut SmallRng::seed_from_u64(seed));
            sb_run(&sb, sweeps, HBSB, true, &mut x, &mut y);
            for (&xi, &yi) in x.iter().zip(y.iter()) {
                prop_assert!(xi.is_finite(), "position not finite: {}", xi);
                prop_assert!(yi.is_finite(), "momentum not finite: {}", yi);
                prop_assert!(xi.abs() <= 1.0, "wall violated: {}", xi);
            }
        }

        /// Hypothesis: the ancilla gauge is a symmetry of the dynamics.
        /// Negating every initial condition negates the whole trajectory
        /// exactly, because every operation in the update is sign-symmetric in
        /// IEEE arithmetic, so the gauge-fixed spins come out identical. This
        /// fails if the gauge fix is dropped or applied to the wrong index.
        #[test]
        fn prop_ancilla_gauge_is_consistent(
            n in 1usize..=5,
            h in prop::collection::vec(-2.0f64..2.0, 1..=5),
            edge_data in prop::collection::vec(
                (any::<u8>(), any::<u8>(), -2.0f64..2.0),
                0..=10,
            ),
            seed in any::<u64>(),
            sweeps in 1usize..=48,
        ) {
            let mut g = graph_from_parts(n, h, edge_data);
            // Force the ancilla to exist; a generated all-zero bias vector would
            // make this property vacuous and the shrinker drives toward zero.
            g.h[0] = 1.0;
            let sb = SbGraph::from_base(&g);
            prop_assert!(sb.has_bias);
            let m = sb.num_particles();
            let (x0, y0) = draw_initial_conditions(m, &mut SmallRng::seed_from_u64(seed));

            let (mut xa, mut ya) = (x0.clone(), y0.clone());
            sb_run(&sb, sweeps, DSB, false, &mut xa, &mut ya);

            let mut xb: Vec<Real> = x0.iter().map(|v| -v).collect();
            let mut yb: Vec<Real> = y0.iter().map(|v| -v).collect();
            sb_run(&sb, sweeps, DSB, false, &mut xb, &mut yb);

            prop_assert_eq!(gauge_fixed_spins(&sb, &xa), gauge_fixed_spins(&sb, &xb));
        }
    }
    /// Hypothesis: the wall invariant survives a long integration under the
    /// strongest heating the shipped variants use.
    ///
    /// `prop_no_nan_and_walls_hold` already checks this for HbSB across random
    /// graphs, but only out to 6 nodes and 64 steps. A wall escape driven by
    /// heating is cumulative: gamma re-injects momentum after every clip, so
    /// the failure mode needs many consecutive clipped steps to show. This runs
    /// one 24-node instance for 2048 steps, thirty-two times the property
    /// test's horizon, and asserts the same invariant at the end.
    ///
    /// A failure means the heating term is applied before the wall clip rather
    /// than after it, or reads the post-collision momentum rather than the
    /// pre-step momentum. Both are kernel defects. Report one rather than
    /// reordering the update here.
    #[test]
    fn hbsb_walls_hold_under_strong_heating() {
        let n = 24usize;
        let mut rng = SmallRng::seed_from_u64(99);
        let h: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..=1.0)).collect();
        let mut edges = Vec::new();
        let mut j = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                if rng.gen_bool(0.2) {
                    edges.push((u, v));
                    j.push(rng.gen_range(-1.0..=1.0));
                }
            }
        }
        let g = IsingGraph::new(h, j, edges);
        let params = sb_params(1, 2048, 21);

        let (x, y) = sb_final_state_for_test(&g, &params, HBSB);
        assert_eq!(x.len(), y.len(), "positions and momenta must match length");
        for (i, &xi) in x.iter().enumerate() {
            assert!(
                xi.is_finite() && xi.abs() <= 1.0,
                "particle {i} escaped the wall under gamma = {}: x = {xi}",
                HBSB.gamma
            );
        }
        assert!(
            y.iter().all(|v| v.is_finite()),
            "heating must not drive a momentum non-finite: {y:?}"
        );
    }
}
