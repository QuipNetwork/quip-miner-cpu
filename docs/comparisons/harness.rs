//! Kernel comparison anchored on the Advantage2-System1 topology.
//!
//! Pivot: `isingmark/fixtures/advantage2-system1.spec.json`, 4577 nodes and
//! 41515 edges (mean degree 18.14). The size ladder expands linearly in both
//! directions from that pivot.
//!
//! - Below the pivot: the induced subgraph on the first `k` hardware node ids,
//!   relabelled dense. This keeps the local Zephyr-style connectivity.
//! - Above the pivot: whole tiles of the pivot, plus random inter-tile edges
//!   chosen to hold the global mean degree at the pivot's value. Disjoint
//!   tiles alone would decompose into independent subproblems and make the
//!   comparison meaningless.
//!
//! Biases are zero everywhere. Couplings are drawn uniformly from the
//! topology's `allowed_j_milli` set, which is {-1000, +1000} for this fixture.

use quip_miner_core::{Algorithm, IsingGraph, SampleParams};
use quip_miner_cpu::gibbs_parallel::{sample_gibbs_with, GibbsConfig};
use quip_miner_cpu::{sample_ising, sample_sb, DSB};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

const FIXTURE: &str =
    "/Users/carback1/Code/quip/isingmark/fixtures/advantage2-system1.spec.json";

const SAMPLES: usize = 30;
const READS: usize = 16;
const SWEEPS: usize = 1_000;

/// The pivot topology as a dense-relabelled edge list.
struct Pivot {
    num_nodes: usize,
    edges: Vec<(usize, usize)>,
    allowed_j_milli: Vec<i64>,
}

fn load_pivot() -> Pivot {
    let raw = std::fs::read_to_string(FIXTURE).expect("read advantage2-system1 fixture");
    let v: serde_json::Value = serde_json::from_str(&raw).expect("parse fixture");

    // Hardware node ids are sparse (yield gaps), so relabel to 0..N-1 in
    // ascending id order. The order matters: the "first k ids" subgraph below
    // the pivot has to be reproducible.
    let ids: Vec<usize> = v["nodes"]
        .as_array()
        .expect("nodes array")
        .iter()
        .map(|x| x.as_u64().expect("node id") as usize)
        .collect();
    let mut index = vec![usize::MAX; ids.iter().copied().max().expect("nonempty") + 1];
    for (dense, &id) in ids.iter().enumerate() {
        index[id] = dense;
    }

    let edges: Vec<(usize, usize)> = v["edges"]
        .as_array()
        .expect("edges array")
        .iter()
        .map(|e| {
            let a = e[0].as_u64().expect("edge u") as usize;
            let b = e[1].as_u64().expect("edge v") as usize;
            (index[a], index[b])
        })
        .collect();

    let allowed_j_milli: Vec<i64> = v["allowed_j_milli"]
        .as_array()
        .expect("allowed_j_milli")
        .iter()
        .map(|x| x.as_i64().expect("j value"))
        .collect();

    Pivot {
        num_nodes: ids.len(),
        edges,
        allowed_j_milli,
    }
}

/// Topology at `target_nodes`, expanded linearly from the pivot.
///
/// Below the pivot the subgraph is grown breadth-first from node 0 rather than
/// taken on the lowest node ids. This hardware graph wires low ids to high ones
/// (edge `[0, 2400]` is typical), so a lowest-id cut keeps barely a fifth of the
/// expected degree and would change density instead of only size.
fn topology_at(p: &Pivot, target_nodes: usize, seed: u64) -> Vec<(usize, usize)> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut edges: Vec<(usize, usize)> = Vec::new();

    if target_nodes < p.num_nodes {
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); p.num_nodes];
        for &(u, v) in &p.edges {
            adj[u].push(v);
            adj[v].push(u);
        }
        let mut keep = vec![false; p.num_nodes];
        let mut order = Vec::with_capacity(target_nodes);
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(0usize);
        keep[0] = true;
        while let Some(x) = queue.pop_front() {
            order.push(x);
            if order.len() == target_nodes {
                break;
            }
            for &y in &adj[x] {
                if !keep[y] {
                    keep[y] = true;
                    queue.push_back(y);
                }
            }
        }
        // Nodes marked but never dequeued are not in the subgraph.
        let mut dense = vec![usize::MAX; p.num_nodes];
        for (i, &x) in order.iter().enumerate() {
            dense[x] = i;
        }
        for &(u, v) in &p.edges {
            if dense[u] != usize::MAX && dense[v] != usize::MAX {
                edges.push((dense[u], dense[v]));
            }
        }
        return edges;
    }

    let tiles = target_nodes.div_ceil(p.num_nodes);
    for t in 0..tiles {
        let base = t * p.num_nodes;
        for &(u, v) in &p.edges {
            let (u, v) = (base + u, base + v);
            if u < target_nodes && v < target_nodes {
                edges.push((u, v));
            }
        }
    }

    // Stitch tiles together so the instance does not decompose. Target the
    // pivot's own edge density; the shortfall after tiling is exactly the
    // edges lost at tile boundaries plus the missing cross-tile connectivity.
    if tiles > 1 {
        let density = p.edges.len() as f64 / p.num_nodes as f64;
        let want = (density * target_nodes as f64).round() as usize;
        while edges.len() < want {
            let u = rng.gen_range(0..target_nodes);
            let v = rng.gen_range(0..target_nodes);
            // Only cross-tile pairs: within-tile structure is already the
            // hardware graph and must not be diluted.
            if u != v && (u / p.num_nodes) != (v / p.num_nodes) {
                edges.push((u.min(v), u.max(v)));
            }
        }
    }
    edges
}

/// One instance: h = 0 everywhere, couplings drawn from the allowed set.
fn instance(edges: &[(usize, usize)], num_nodes: usize, allowed: &[i64], seed: u64) -> IsingGraph {
    let mut rng = SmallRng::seed_from_u64(seed);
    let h = vec![0.0f64; num_nodes];
    let j: Vec<f64> = (0..edges.len())
        .map(|_| allowed[rng.gen_range(0..allowed.len())] as f64 / 1000.0)
        .collect();
    IsingGraph::new(h, j, edges.to_vec())
}

#[derive(Clone, Copy)]
enum Kernel {
    Sa,
    Gibbs,
    Sb,
}

impl Kernel {
    fn name(self) -> &'static str {
        match self {
            Self::Sa => "cpu-sa",
            Self::Gibbs => "cpu-gibbs",
            Self::Sb => "cpu-sb",
        }
    }

    fn best(self, g: &IsingGraph, p: &SampleParams) -> i64 {
        let out = match self {
            Self::Sa => sample_ising(g, p, Algorithm::Sa),
            Self::Gibbs => sample_gibbs_with(g, p, &GibbsConfig::default()).expect("default config"),
            Self::Sb => sample_sb(g, p, DSB),
        };
        out.iter()
            .map(|r| r.energy_milli)
            .min()
            .expect("num_reads > 0")
    }
}

const KERNELS: [Kernel; 3] = [Kernel::Sa, Kernel::Gibbs, Kernel::Sb];

/// One-minute load average, so a contaminated row is visible in the data.
fn load_avg() -> f64 {
    std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| {
            s.trim()
                .trim_matches(|c| c == '{' || c == '}' || c == ' ')
                .split_whitespace()
                .next()
                .and_then(|v| v.parse::<f64>().ok())
        })
        .unwrap_or(f64::NAN)
}

fn main() {
    let p = load_pivot();
    eprintln!(
        "pivot: {} nodes, {} edges, mean degree {:.2}",
        p.num_nodes,
        p.edges.len(),
        2.0 * p.edges.len() as f64 / p.num_nodes as f64
    );

    // Linear ladder in quarter-pivot steps, pivot at 1.00.
    let all = [0.25f64, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00];
    let only_small = std::env::args().any(|a| a == "--small");
    let factors: Vec<f64> = all
        .into_iter()
        .filter(|f| !only_small || *f < 1.0)
        .collect();

    let mut rows: Vec<serde_json::Value> = Vec::new();
    println!("| kernel | nodes | edges | cores | reads | sweeps | best energy (30 samples) | mean time/sample | load |");
    println!("|--------|------:|------:|------:|------:|-------:|-------------------------:|-----------------:|-----:|");

    for f in factors {
        let nodes = ((p.num_nodes as f64) * f).round() as usize;
        let edges = topology_at(&p, nodes, 20_260_807);
        // One instance per sample index, shared across kernels so every kernel
        // sees the identical 30 problems.
        let instances: Vec<IsingGraph> = (0..SAMPLES)
            .map(|s| instance(&edges, nodes, &p.allowed_j_milli, 900 + s as u64))
            .collect();

        for k in KERNELS {
            let mut best = i64::MAX;
            let mut total = Duration::ZERO;
            let load_start = load_avg();
            let mut per_sample_best: Vec<i64> = Vec::with_capacity(SAMPLES);
            for (s, g) in instances.iter().enumerate() {
                let params = SampleParams {
                    num_reads: READS,
                    num_sweeps: SWEEPS,
                    seed: 4_000 + s as u64,
                    ..Default::default()
                };
                let t0 = Instant::now();
                let e = k.best(g, &params);
                total += t0.elapsed();
                per_sample_best.push(e);
                best = best.min(e);
            }
            let mean_ms = total.as_secs_f64() * 1e3 / SAMPLES as f64;
            let mean_energy =
                per_sample_best.iter().sum::<i64>() as f64 / SAMPLES as f64;
            let load = (load_start + load_avg()) / 2.0;
            let cores = if matches!(k, Kernel::Gibbs) {
                GibbsConfig::default().workers
            } else {
                1
            };
            println!(
                "| {} | {nodes} | {} | {cores} | {READS} | {SWEEPS} | {best} | {mean_ms:.1} ms | {load:.1} |",
                k.name(),
                edges.len()
            );
            rows.push(serde_json::json!({
                "kernel": k.name(),
                "cores": if matches!(k, Kernel::Gibbs) { GibbsConfig::default().workers } else { 1 },
                "factor": f,
                "nodes": nodes,
                "edges": edges.len(),
                "reads": READS,
                "sweeps": SWEEPS,
                "samples": SAMPLES,
                "best_energy_milli": best,
                "mean_best_energy_milli": mean_energy,
                "per_sample_best_milli": per_sample_best,
                "mean_time_ms": mean_ms,
                "load_avg": (load_start + load_avg()) / 2.0,
            }));
            eprintln!("done {} n={nodes}", k.name());
        }
    }

    let out = serde_json::json!({
        "pivot": {
            "name": "advantage2-system1",
            "nodes": p.num_nodes,
            "edges": p.edges.len(),
        },
        "bias": "h = 0 (couplings only)",
        "allowed_j_milli": p.allowed_j_milli,
        "rows": rows,
    });
    let out_path = if only_small { "results_small.json" } else { "results.json" };
    std::fs::write(
        out_path,
        serde_json::to_string_pretty(&out).expect("serialize"),
    )
    .expect("write results.json");
    eprintln!("wrote results.json");
}
