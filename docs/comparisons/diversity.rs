//! Solution diversity at the pivot, per kernel.
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
use quip_miner_cpu::{sample_ising, sample_ising_mps, sample_sb, InitMode, MpsConfig, BSB, DSB};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;

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
    Bsb,
    Mps,
    Mfa,
}

/// Bond cap for the tensor-network arms. `Mps` gets the binary's ceiling of 32,
/// which the per-job budget lowers on its own; `Mfa` is pinned to 1 by
/// definition. `time_budget_ms: 0` disables the wall-clock valve so a loaded
/// host cannot change the answer, only the time it takes to get it.
fn mps_cfg(chi_max: usize) -> MpsConfig {
    MpsConfig {
        chi_max,
        init: InitMode::Anneal,
        time_budget_ms: 0,
        flop_budget: 1.25e9,
    }
}

impl Kernel {
    fn name(self) -> &'static str {
        match self {
            Self::Sa => "cpu-sa",
            Self::Gibbs => "cpu-gibbs",
            Self::Sb => "cpu-sb",
            Self::Bsb => "cpu-bsb",
            Self::Mps => "cpu-mps",
            Self::Mfa => "cpu-mfa",
        }
    }

    fn reads(self, g: &IsingGraph, p: &SampleParams) -> Vec<quip_miner_core::SamplerResult> {
        match self {
            Self::Sa => sample_ising(g, p, Algorithm::Sa),
            Self::Gibbs => sample_gibbs_with(g, p, &GibbsConfig::default()).expect("default config"),
            Self::Sb => sample_sb(g, p, DSB),
            Self::Bsb => sample_sb(g, p, BSB),
            Self::Mps => sample_ising_mps(g, p, &mps_cfg(32)),
            Self::Mfa => sample_ising_mps(g, p, &mps_cfg(1)),
        }
    }
}

const KERNELS: [Kernel; 6] = [
    Kernel::Sa,
    Kernel::Gibbs,
    Kernel::Sb,
    Kernel::Bsb,
    Kernel::Mps,
    Kernel::Mfa,
];

/// Diversity is what a coordinator actually consumes: a batch of identical
/// reads is worth one read no matter how good that read is. Measured three
/// ways, all over the same 16 reads that the ladder scores:
///
/// - distinct spin configurations, the raw count of different answers;
/// - distinct energies, which ignores symmetry-equivalent relabellings;
/// - the spread between the best and the median read, in energy.
///
/// Every number here is deterministic given the seeds, so host load cannot
/// contaminate this table the way it contaminates a timing table.
fn main() {
    let p = load_pivot();
    let edges = topology_at(&p, p.num_nodes, 20_260_807);
    let instances: Vec<IsingGraph> = (0..SAMPLES)
        .map(|s| instance(&edges, p.num_nodes, &p.allowed_j_milli, 900 + s as u64))
        .collect();

    println!("| kernel | distinct spins / 16 | distinct energies / 16 | best | median | spread |");
    println!("|--------|--------------------:|-----------------------:|-----:|-------:|-------:|");

    let mut rows: Vec<serde_json::Value> = Vec::new();
    for k in KERNELS {
        let mut spin_sets = 0usize;
        let mut energy_sets = 0usize;
        let mut best_sum = 0i64;
        let mut median_sum = 0i64;
        for (s, g) in instances.iter().enumerate() {
            let params = SampleParams {
                num_reads: READS,
                num_sweeps: SWEEPS,
                seed: 4_000 + s as u64,
                ..Default::default()
            };
            let out = k.reads(g, &params);
            spin_sets += out
                .iter()
                .map(|r| r.spins.clone())
                .collect::<HashSet<_>>()
                .len();
            let mut es: Vec<i64> = out.iter().map(|r| r.energy_milli).collect();
            energy_sets += es.iter().copied().collect::<HashSet<_>>().len();
            es.sort_unstable();
            best_sum += es[0];
            median_sum += es[es.len() / 2];
        }
        let n = SAMPLES as f64;
        let (ds, de) = (spin_sets as f64 / n, energy_sets as f64 / n);
        let (best, median) = (best_sum as f64 / n, median_sum as f64 / n);
        println!(
            "| {} | {ds:.2} | {de:.2} | {best:.0} | {median:.0} | {:.0} |",
            k.name(),
            median - best
        );
        rows.push(serde_json::json!({
            "kernel": k.name(),
            "distinct_spins": ds,
            "distinct_energies": de,
            "mean_best_milli": best,
            "mean_median_milli": median,
        }));
    }
    std::fs::write(
        "diversity.json",
        serde_json::to_string_pretty(&serde_json::json!({
            "nodes": p.num_nodes, "edges": edges.len(),
            "reads": READS, "sweeps": SWEEPS, "samples": SAMPLES,
            "rows": rows,
        }))
        .expect("serialize"),
    )
    .expect("write diversity.json");
}
