//! Parallel chromatic Gibbs.
//!
//! One sweep visits every colour class in order. A class is an independent set,
//! so every member's conditional distribution depends only on nodes outside the
//! class, and the whole class resamples at once. Classes run in sequence,
//! because class `c + 1` reads what class `c` wrote. Workers split the members
//! of a single class, which is why the worker count does not have to match the
//! class count.
//!
//! Two properties the sequential scan did not have to keep:
//!
//! - Fields are recomputed per class rather than patched. The sequential kernel
//!   patches a shared `heff` after every flip, which several workers cannot do
//!   at once without racing. Recomputing costs the same `2E` per sweep.
//! - Randomness is counter-based, keyed on read, sweep, class and node. A
//!   per-worker generator would make the output depend on the worker count, and
//!   the protocol requires that one seed and one binary reproduce one result on
//!   any machine.

use std::sync::atomic::{AtomicBool, AtomicI8, AtomicUsize, Ordering};

use quip_miner_core::{IsingGraph, SampleParams, SamplerResult};
use quip_protocol::scoring::energy_milli;

use crate::coloring::Coloring;
use crate::sampler_core::CpuGraph;

/// Default workers per model. Measured as the best setting for the
/// Advantage2-System1 topology; see `docs/comparisons.md`.
pub const DEFAULT_GIBBS_WORKERS: usize = 4;

/// SplitMix64. Counter-based, so a draw depends only on its key and never on
/// how the work was divided between workers.
#[inline]
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Uniform draw in `[0, 1)` for one node at one point in the schedule.
#[inline]
fn draw_u01(seed: u64, step: u64, node: u32) -> f64 {
    let key = seed
        ^ splitmix64(step.wrapping_mul(0x2545_F491_4F6C_DD1D))
        ^ splitmix64(u64::from(node).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    // 53 bits into the mantissa, matching the usual f64 uniform construction.
    (splitmix64(key) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Heat-bath conditional for one node given its effective field.
#[inline]
fn heat_bath_spin(heff: f64, beta: f64, u: f64) -> i8 {
    let arg = (2.0 * beta * heff).clamp(-500.0, 500.0);
    let p_plus = 1.0 / (1.0 + arg.exp());
    if u < p_plus {
        1
    } else {
        -1
    }
}

/// Effective field at `var` from the current spins.
///
/// Read-only over `spins`, so every worker in a class can call it at once.
#[inline]
fn field_at(graph: &CpuGraph, spins: &[AtomicI8], var: usize) -> f64 {
    let (nodes, coups) = graph.neighbors(var);
    let mut acc = graph.bias(var);
    for (&w, &j) in nodes.iter().zip(coups.iter()) {
        acc += j * f64::from(spins[w as usize].load(Ordering::Relaxed));
    }
    acc
}

/// Run one read of chromatic Gibbs over `beta_schedule`, returning the spins.
///
/// `workers` splits each colour class. The result does not depend on it.
pub(crate) fn gibbs_read(
    graph: &CpuGraph,
    coloring: &Coloring,
    beta_schedule: &[f64],
    sweeps_per_beta: usize,
    seed: u64,
    workers: usize,
    spins: &[AtomicI8],
) {
    let classes = coloring.classes();
    if classes.is_empty() {
        return;
    }

    // One worker needs no pool and no barriers. Measured at the pivot topology,
    // spawning per class cost more than the class update itself: 8 classes over
    // 1000 sweeps is 8000 spawn-and-join cycles per read, and throughput fell
    // as workers rose. The pool below spawns once per read instead, and
    // synchronises on a barrier at each class boundary.
    if workers <= 1 {
        let mut step: u64 = 0;
        for &beta in beta_schedule {
            for _ in 0..sweeps_per_beta {
                for class in classes {
                    update_range(graph, spins, class, beta, seed, step);
                    step += 1;
                }
            }
        }
        return;
    }

    let barrier = SpinBarrier::new(workers);
    std::thread::scope(|scope| {
        for wid in 0..workers {
            let barrier = &barrier;
            scope.spawn(move || {
                let mut step: u64 = 0;
                let mut sense = false;
                for &beta in beta_schedule {
                    for _ in 0..sweeps_per_beta {
                        for class in classes {
                            let chunk = class.len().div_ceil(workers);
                            let start = (wid * chunk).min(class.len());
                            let end = (start + chunk).min(class.len());
                            update_range(graph, spins, &class[start..end], beta, seed, step);
                            step += 1;
                            // Class `c + 1` reads what class `c` wrote, so every
                            // worker must finish this class before any worker
                            // starts the next one.
                            barrier.wait(&mut sense);
                        }
                    }
                }
            });
        }
    });
}

/// Sense-reversing spin barrier.
///
/// A class update is one or two microseconds of work per worker, and a sweep
/// crosses one barrier per class. A mutex-and-condvar barrier costs more than
/// the work it separates at that granularity, so this spins instead.
struct SpinBarrier {
    waiting: AtomicUsize,
    sense: AtomicBool,
    workers: usize,
}

impl SpinBarrier {
    fn new(workers: usize) -> Self {
        Self {
            waiting: AtomicUsize::new(0),
            sense: AtomicBool::new(false),
            workers,
        }
    }

    /// `local` carries this worker's expected sense and flips on every call.
    fn wait(&self, local: &mut bool) {
        *local = !*local;
        if self.waiting.fetch_add(1, Ordering::AcqRel) + 1 == self.workers {
            self.waiting.store(0, Ordering::Release);
            self.sense.store(*local, Ordering::Release);
        } else {
            // Spin briefly, then yield. A pure spin collapses under
            // oversubscription: spinning workers hold cores that runnable
            // workers need, and the barrier never completes on time. Measured
            // at 16 workers on a 12-core host, a pure spin ran roughly 90 times
            // slower than one worker.
            let mut spins = 0u32;
            while self.sense.load(Ordering::Acquire) != *local {
                if spins < 512 {
                    std::hint::spin_loop();
                    spins += 1;
                } else {
                    std::thread::yield_now();
                }
            }
        }
    }
}

/// Resample every node in `part`, which must lie inside one colour class.
#[inline]
fn update_range(
    graph: &CpuGraph,
    spins: &[AtomicI8],
    part: &[u32],
    beta: f64,
    seed: u64,
    step: u64,
) {
    for &v in part {
        let h = field_at(graph, spins, v as usize);
        let u = draw_u01(seed, step, v);
        spins[v as usize].store(heat_bath_spin(h, beta, u), Ordering::Relaxed);
    }
}

/// Sample `num_reads` independent chromatic-Gibbs reads.
///
/// `workers` splits each colour class. It changes throughput only: the returned
/// spins are identical at any worker count.
///
/// # Examples
///
/// ```
/// use quip_miner_cpu::gibbs_parallel::sample_gibbs_parallel;
/// use quip_miner_cpu::{IsingGraph, SampleParams};
///
/// let graph = IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)]);
/// let params = SampleParams { num_reads: 2, num_sweeps: 64, seed: 1, ..Default::default() };
/// let results = sample_gibbs_parallel(&graph, &params, 4);
/// assert_eq!(results.len(), 2);
/// ```
pub fn sample_gibbs_parallel(
    graph: &IsingGraph,
    params: &SampleParams,
    workers: usize,
) -> Vec<SamplerResult> {
    let cpu = CpuGraph::from_base(graph);
    let coloring = Coloring::new(&cpu);
    let betas = crate::sampler_core::build_beta_schedule(graph, params);
    let sweeps_per_beta = params.sweeps_per_beta.max(1);
    let n = cpu.num_nodes();
    let num_reads = params.num_reads.max(1);

    (0..num_reads)
        .map(|read_idx| {
            let seed = params
                .seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(read_idx as u64)
                .wrapping_add(1);
            // Initial spins come from the same counter-based stream as the
            // sweeps, so the starting point is worker-count independent too.
            let spins: Vec<AtomicI8> = (0..n)
                .map(|v| {
                    let u = draw_u01(seed, u64::MAX, v as u32);
                    AtomicI8::new(if u < 0.5 { 1 } else { -1 })
                })
                .collect();
            gibbs_read(&cpu, &coloring, &betas, sweeps_per_beta, seed, workers, &spins);
            let out: Vec<i8> = spins.iter().map(|a| a.load(Ordering::Relaxed)).collect();
            SamplerResult {
                energy_milli: energy_milli(&out, &graph.h, &graph.j, &graph.edges),
                spins: out,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler_core::CpuGraph;
    use quip_miner_core::IsingGraph;

    fn atomics(vals: &[i8]) -> Vec<AtomicI8> {
        vals.iter().map(|&v| AtomicI8::new(v)).collect()
    }

    fn read_back(s: &[AtomicI8]) -> Vec<i8> {
        s.iter().map(|a| a.load(Ordering::Relaxed)).collect()
    }

    fn ferro_ring(n: usize) -> IsingGraph {
        let edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
        IsingGraph::new(vec![0.0; n], vec![-1.0; n], edges)
    }

    /// Hypothesis: the result does not depend on the worker count. This is the
    /// property that lets an operator tune workers for throughput without
    /// changing what the miner submits. Without it, a machine with a different
    /// core count would produce a different answer from the same seed.
    #[test]
    fn output_is_independent_of_worker_count() {
        let g = ferro_ring(64);
        let cpu = CpuGraph::from_base(&g);
        let coloring = Coloring::new(&cpu);
        let betas = vec![0.1, 0.5, 1.0, 2.0];

        let start: Vec<i8> = (0..64).map(|i| if i % 3 == 0 { 1 } else { -1 }).collect();
        let mut reference = None;
        for workers in [1usize, 2, 3, 4, 8, 16] {
            let spins = atomics(&start);
            gibbs_read(&cpu, &coloring, &betas, 4, 99, workers, &spins);
            let got = read_back(&spins);
            match &reference {
                None => reference = Some(got),
                Some(want) => assert_eq!(
                    &got, want,
                    "worker count {workers} changed the result"
                ),
            }
        }
    }

    /// Hypothesis: the kernel only ever writes valid wire spins.
    #[test]
    fn every_spin_stays_pm1() {
        let g = ferro_ring(32);
        let cpu = CpuGraph::from_base(&g);
        let coloring = Coloring::new(&cpu);
        let spins = atomics(&[1i8; 32]);
        gibbs_read(&cpu, &coloring, &[0.5, 1.5], 3, 7, 4, &spins);
        assert!(read_back(&spins).iter().all(|&s| s == 1 || s == -1));
    }

    /// Hypothesis: at a high beta the sampler drives a ferromagnetic ring to a
    /// nearly aligned state. This is the check that the conditional carries the
    /// right sign: an inverted heat-bath probability would anti-align it.
    #[test]
    fn cold_ferromagnet_aligns() {
        let g = ferro_ring(128);
        let cpu = CpuGraph::from_base(&g);
        let coloring = Coloring::new(&cpu);
        let spins = atomics(&[1i8; 128]);
        gibbs_read(&cpu, &coloring, &[8.0], 60, 5, 4, &spins);
        let out = read_back(&spins);
        let up = out.iter().filter(|&&s| s == 1).count();
        let aligned = up.max(128 - up);
        assert!(aligned >= 120, "expected near alignment, got {up} up of 128");
    }

    /// Hypothesis: a positive bias with no couplings drives the spin negative,
    /// because the miner minimizes `h_i s_i`.
    #[test]
    fn positive_bias_drives_spin_negative() {
        let g = IsingGraph::new(vec![2.0], vec![], vec![]);
        let cpu = CpuGraph::from_base(&g);
        let coloring = Coloring::new(&cpu);
        let spins = atomics(&[1]);
        gibbs_read(&cpu, &coloring, &[6.0], 20, 11, 2, &spins);
        assert_eq!(read_back(&spins), vec![-1]);
    }

    /// Hypothesis: an empty graph has no classes, so the kernel does nothing
    /// and does not panic.
    #[test]
    fn empty_graph_is_a_no_op() {
        let g = IsingGraph::new(vec![], vec![], vec![]);
        let cpu = CpuGraph::from_base(&g);
        let coloring = Coloring::new(&cpu);
        let spins: Vec<AtomicI8> = Vec::new();
        gibbs_read(&cpu, &coloring, &[1.0], 2, 3, 4, &spins);
        assert!(read_back(&spins).is_empty());
    }
}
