//! Reproducible CPU multi-spin annealing measurement harness.

use clap::Parser;
use quip_miner_cpu::{IsingGraph, SaSampler, SaVariant, SampleParams};
use quip_solver_core::{Sampler, SamplerResult};
use std::collections::BTreeSet;
use std::fmt;
use std::time::Instant;

const SEED: u64 = 19;
const SWEEPS_PER_BETA: usize = 4;

#[derive(Parser)]
#[command(about = "Reproducible CPU MSA timing harness")]
struct Args {
    /// Timed repeats per case after one warmup.
    #[arg(long, default_value_t = 5)]
    repeats: usize,
    /// Smaller smoke matrix: shrink mixed128 and omit setup128 and setup128_fresh.
    #[arg(long)]
    quick: bool,
}

#[derive(Debug)]
enum BenchError {
    Repeats,
    Graph(&'static str),
    Sample(String),
    Fingerprint(&'static str),
}

impl fmt::Display for BenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Repeats => write!(f, "repeats must be a positive integer"),
            Self::Graph(msg) => write!(f, "{msg}"),
            Self::Sample(msg) => write!(f, "sample failed: {msg}"),
            Self::Fingerprint(case) => {
                write!(f, "fingerprint mismatch for {case}")
            }
        }
    }
}

impl std::error::Error for BenchError {}

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    nodes: usize,
    degree: usize,
    reads: usize,
    sweeps: usize,
    beta: (f64, f64),
    fresh_sampler: bool,
}

struct CaseResult {
    name: &'static str,
    nodes: usize,
    degree: usize,
    reads: usize,
    sweeps: usize,
    median_ns: u128,
    fingerprint: String,
    best_energy_milli: i64,
    distinct_solution_count: usize,
}

fn ring_graph(nodes: usize, degree: usize) -> Result<IsingGraph, &'static str> {
    if !degree.is_multiple_of(2) {
        return Err("degree must be even");
    }
    if nodes <= degree {
        return Err("nodes must exceed degree");
    }
    let mut edges = Vec::new();
    let mut couplings = Vec::new();
    for u in 0..nodes {
        for offset in 1..=degree / 2 {
            edges.push((u, (u + offset) % nodes));
            couplings.push(if (u + offset) % 3 == 0 { -1.0 } else { 1.0 });
        }
    }
    let h: Vec<f64> = (0..nodes).map(|u| (u % 3) as f64 - 1.0).collect();
    Ok(IsingGraph::new(h, couplings, edges))
}

fn fingerprint(results: &[SamplerResult]) -> String {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0100_0000_01b3;
    let mut hash = OFFSET;
    let mut absorb = |byte: u8| {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    };
    for result in results {
        for byte in result.energy_milli.to_le_bytes() {
            absorb(byte);
        }
        for &spin in &result.spins {
            absorb(spin as u8);
        }
    }
    format!("{hash:016x}")
}

fn median_ns(samples: &mut [u128]) -> u128 {
    samples.sort_unstable();
    let n = samples.len();
    if n.is_multiple_of(2) {
        if n == 0 {
            0
        } else {
            (samples[n / 2 - 1] + samples[n / 2]) / 2
        }
    } else {
        samples[n / 2]
    }
}

fn summarize(results: &[SamplerResult]) -> (i64, usize) {
    let best = results.iter().map(|r| r.energy_milli).min().unwrap_or(0);
    let distinct: BTreeSet<&[i8]> = results.iter().map(|r| r.spins.as_slice()).collect();
    (best, distinct.len())
}

fn cases(quick: bool) -> Vec<Case> {
    let mixed_nodes = if quick { 512 } else { 4096 };
    let mixed_sweeps = if quick { 64 } else { 256 };
    let mut out = vec![
        Case {
            name: "hot64",
            nodes: 512,
            degree: 20,
            reads: 64,
            sweeps: 128,
            beta: (0.005, 0.02),
            fresh_sampler: false,
        },
        Case {
            name: "mixed128",
            nodes: mixed_nodes,
            degree: 20,
            reads: 128,
            sweeps: mixed_sweeps,
            beta: (0.05, 4.0),
            fresh_sampler: false,
        },
        Case {
            name: "cold128",
            nodes: 512,
            degree: 20,
            reads: 128,
            sweeps: 128,
            beta: (2.0, 8.0),
            fresh_sampler: false,
        },
        Case {
            name: "sparse64",
            nodes: 512,
            degree: 6,
            reads: 64,
            sweeps: 128,
            beta: (0.05, 4.0),
            fresh_sampler: false,
        },
        Case {
            name: "dense64",
            nodes: 256,
            degree: 48,
            reads: 64,
            sweeps: 128,
            beta: (0.05, 4.0),
            fresh_sampler: false,
        },
    ];
    if !quick {
        out.push(Case {
            name: "setup128",
            nodes: 4096,
            degree: 20,
            reads: 128,
            sweeps: 4,
            beta: (0.05, 4.0),
            fresh_sampler: false,
        });
        out.push(Case {
            name: "setup128_fresh",
            nodes: 4096,
            degree: 20,
            reads: 128,
            sweeps: 4,
            beta: (0.05, 4.0),
            fresh_sampler: true,
        });
    }
    out
}

fn run_sample(
    sampler: &SaSampler,
    graph: &IsingGraph,
    params: &SampleParams,
) -> Result<(u128, Vec<SamplerResult>), BenchError> {
    let started = Instant::now();
    let results = sampler
        .sample(graph, params)
        .map_err(|e| BenchError::Sample(e.to_string()))?;
    Ok((started.elapsed().as_nanos(), results))
}

fn measure_case(case: Case, repeats: usize) -> Result<CaseResult, BenchError> {
    let graph = ring_graph(case.nodes, case.degree).map_err(BenchError::Graph)?;
    let params = SampleParams {
        num_reads: case.reads,
        num_sweeps: case.sweeps,
        sweeps_per_beta: SWEEPS_PER_BETA,
        beta_range: Some(case.beta),
        seed: SEED,
    };

    let warmup_sampler = SaSampler::new(SaVariant::MultiSpin);
    let warmup = warmup_sampler
        .sample(&graph, &params)
        .map_err(|e| BenchError::Sample(e.to_string()))?;
    let expected_fp = fingerprint(&warmup);
    let (best_energy_milli, distinct_solution_count) = summarize(&warmup);

    let mut times = Vec::with_capacity(repeats);
    if case.fresh_sampler {
        for _ in 0..repeats {
            let sampler = SaSampler::new(SaVariant::MultiSpin);
            let (elapsed, results) = run_sample(&sampler, &graph, &params)?;
            if fingerprint(&results) != expected_fp {
                return Err(BenchError::Fingerprint(case.name));
            }
            times.push(elapsed);
        }
    } else {
        for _ in 0..repeats {
            let (elapsed, results) = run_sample(&warmup_sampler, &graph, &params)?;
            if fingerprint(&results) != expected_fp {
                return Err(BenchError::Fingerprint(case.name));
            }
            times.push(elapsed);
        }
    }

    Ok(CaseResult {
        name: case.name,
        nodes: case.nodes,
        degree: case.degree,
        reads: case.reads,
        sweeps: case.sweeps,
        median_ns: median_ns(&mut times),
        fingerprint: expected_fp,
        best_energy_milli,
        distinct_solution_count,
    })
}

#[expect(
    clippy::print_stdout,
    reason = "msa_bench reports one parseable CSV row per case on stdout"
)]
fn emit_csv(results: &[CaseResult]) {
    println!(
        "case,nodes,degree,reads,sweeps,median_ns,fingerprint,best_energy_milli,distinct_solution_count"
    );
    for row in results {
        println!(
            "{},{},{},{},{},{},{},{},{}",
            row.name,
            row.nodes,
            row.degree,
            row.reads,
            row.sweeps,
            row.median_ns,
            row.fingerprint,
            row.best_energy_milli,
            row.distinct_solution_count
        );
    }
}

fn main() -> Result<(), BenchError> {
    let args = Args::parse();
    if args.repeats == 0 {
        return Err(BenchError::Repeats);
    }
    let mut rows = Vec::new();
    for case in cases(args.quick) {
        rows.push(measure_case(case, args.repeats)?);
    }
    emit_csv(&rows);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{cases, fingerprint, ring_graph};
    use quip_solver_core::SamplerResult;
    use std::collections::HashSet;

    fn undirected_key(u: usize, v: usize) -> (usize, usize) {
        (u.min(v), u.max(v))
    }

    #[test]
    fn ring_graph_rejects_odd_degree_and_nodes_not_exceeding_degree() {
        assert!(ring_graph(8, 3).is_err());
        assert!(ring_graph(4, 4).is_err());
        assert!(ring_graph(4, 6).is_err());
    }

    #[test]
    fn ring_graph_degree_is_regular() {
        for (nodes, degree) in [(16, 4), (512, 6), (512, 20), (256, 48)] {
            let graph = ring_graph(nodes, degree).expect("valid ring");
            assert_eq!(graph.num_nodes(), nodes);
            assert_eq!(graph.h.len(), nodes);
            assert!(graph.h.iter().all(|&h| h == -1.0 || h == 0.0 || h == 1.0));
            let mut adj = vec![HashSet::new(); nodes];
            for &(u, v) in &graph.edges {
                assert_ne!(u, v, "self-loop at {u}");
                assert!(adj[u].insert(v), "duplicate neighbor {v} of {u}");
                assert!(adj[v].insert(u), "duplicate neighbor {u} of {v}");
            }
            for (u, neighbors) in adj.iter().enumerate() {
                assert_eq!(
                    neighbors.len(),
                    degree,
                    "node {u} degree {} != {degree}",
                    neighbors.len()
                );
            }
        }
    }

    #[test]
    fn ring_graph_edges_are_unique_undirected() {
        for (nodes, degree) in [(16, 4), (512, 6), (512, 20), (256, 48)] {
            let graph = ring_graph(nodes, degree).expect("valid ring");
            let mut unique = HashSet::new();
            for &(u, v) in &graph.edges {
                assert!(
                    unique.insert(undirected_key(u, v)),
                    "duplicate undirected edge ({u},{v})"
                );
            }
            assert_eq!(unique.len(), graph.edges.len());
            assert_eq!(graph.edges.len(), nodes * degree / 2);
            assert_eq!(graph.j.len(), graph.edges.len());
            assert!(graph.j.iter().all(|&j| j == 1.0 || j == -1.0));
        }
    }

    #[test]
    fn fingerprint_changes_when_a_spin_or_energy_changes() {
        let base = vec![SamplerResult {
            spins: vec![1, -1, 1, -1],
            energy_milli: -4,
        }];
        let spin_flip = vec![SamplerResult {
            spins: vec![1, 1, 1, -1],
            energy_milli: -4,
        }];
        let energy_shift = vec![SamplerResult {
            spins: vec![1, -1, 1, -1],
            energy_milli: -3,
        }];
        let same = vec![SamplerResult {
            spins: vec![1, -1, 1, -1],
            energy_milli: -4,
        }];
        assert_eq!(fingerprint(&base), fingerprint(&same));
        assert_ne!(fingerprint(&base), fingerprint(&spin_flip));
        assert_ne!(fingerprint(&base), fingerprint(&energy_shift));
        assert_ne!(fingerprint(&spin_flip), fingerprint(&energy_shift));

        let first = SamplerResult {
            spins: vec![1, -1, 1, -1],
            energy_milli: -4,
        };
        let second = SamplerResult {
            spins: vec![-1, 1, -1, 1],
            energy_milli: 2,
        };
        let two = vec![first.clone(), second.clone()];
        let second_spin = vec![
            first.clone(),
            SamplerResult {
                spins: vec![-1, -1, -1, 1],
                energy_milli: 2,
            },
        ];
        let second_energy = vec![
            first,
            SamplerResult {
                spins: vec![-1, 1, -1, 1],
                energy_milli: 3,
            },
        ];
        assert_ne!(fingerprint(&two), fingerprint(&second_spin));
        assert_ne!(fingerprint(&two), fingerprint(&second_energy));
        assert_ne!(fingerprint(&second_spin), fingerprint(&second_energy));
    }

    #[test]
    fn setup128_pairs_warm_and_fresh_and_quick_omits_both() {
        let full = cases(false);
        let setup = full
            .iter()
            .find(|c| c.name == "setup128")
            .expect("setup128");
        let setup_fresh = full
            .iter()
            .find(|c| c.name == "setup128_fresh")
            .expect("setup128_fresh");
        assert!(!setup.fresh_sampler);
        assert!(setup_fresh.fresh_sampler);
        assert_eq!(setup.nodes, setup_fresh.nodes);
        assert_eq!(setup.degree, setup_fresh.degree);
        assert_eq!(setup.reads, setup_fresh.reads);
        assert_eq!(setup.sweeps, setup_fresh.sweeps);
        assert_eq!(setup.beta, setup_fresh.beta);
        assert_eq!(setup.nodes, 4096);
        assert_eq!(setup.degree, 20);
        assert_eq!(setup.reads, 128);
        assert_eq!(setup.sweeps, 4);
        assert_eq!(setup.beta, (0.05, 4.0));
        let quick = cases(true);
        assert!(quick
            .iter()
            .all(|c| c.name != "setup128" && c.name != "setup128_fresh"));
    }
}
