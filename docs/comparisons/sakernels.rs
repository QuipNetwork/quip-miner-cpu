// Kernel-level rates for the three annealing kernels: the f64 Metropolis
// kernel, the exact integer kernel `cpu-sa` now uses on discrete problems, and
// the tabulated-threshold and multi-spin kernels behind `cpu-fsa` and
// `cpu-msa`.
//
// Not part of the crate build. To run it, copy this file to
// `src/bin/sakernels.rs`, make `sa_int` and `sa_msc` `pub mod` in `src/lib.rs`,
// widen their `pub(crate)` items to `pub`, move `serde_json` from
// `[dev-dependencies]` to `[dependencies]`, and `cargo run --release
// --all-features --bin sakernels`. Point QUIP_TOPOLOGY_SPEC at
// isingmark/fixtures/chain-h0.spec.json to measure on the real Zephyr
// adjacency; without it the first row falls back to a random graph of the same
// size and degree.
//
// Reported figures are spin updates per second. For the multi-spin kernel one
// word update is 64 spin updates, and the last column converts that into the
// figure a 36-read job actually sees, where 28 of the 64 lanes are idle.

use quip_miner_cpu::sa_int::{
    acceptance_table, anneal_from, anneal_one_read_int, sweep_offsets, threshold_draws, IntGraph,
};
use quip_miner_cpu::sa_msc::{anneal_word, bond_counts, MscState, LANES};
use quip_miner_cpu::sampler_core::{anneal_one_read, random_spins, CpuGraph};
use quip_solver_core::beta::geometric_beta_schedule;
use quip_solver_core::IsingGraph;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn random_edges(n: usize, per_node: usize) -> (usize, Vec<(usize, usize)>) {
    let mut rng = SmallRng::seed_from_u64(3);
    let mut edges = Vec::new();
    for _ in 0..(per_node * n) {
        let u = rng.gen_range(0..n);
        let v = rng.gen_range(0..n);
        if u != v {
            edges.push((u, v));
        }
    }
    (n, edges)
}

fn zephyr_or_random(n: usize, per_node: usize) -> (usize, Vec<(usize, usize)>) {
    let Ok(path) = std::env::var("QUIP_TOPOLOGY_SPEC") else {
        return random_edges(n, per_node);
    };
    let Ok(text) = std::fs::read_to_string(path) else {
        return random_edges(n, per_node);
    };
    let spec: serde_json::Value = serde_json::from_str(&text).unwrap();
    let nodes: Vec<usize> = spec["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let mut dense = std::collections::HashMap::new();
    for (i, &id) in nodes.iter().enumerate() {
        dense.insert(id, i);
    }
    let edges = spec["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            (
                dense[&(e[0].as_u64().unwrap() as usize)],
                dense[&(e[1].as_u64().unwrap() as usize)],
            )
        })
        .collect();
    (nodes.len(), edges)
}

fn main() {
    for (name, (n, edges)) in [
        ("zephyr", zephyr_or_random(4577, 9)),
        ("random", random_edges(4577, 9)),
        ("small", random_edges(512, 3)),
    ] {
        let mut rng = SmallRng::seed_from_u64(7);
        let j: Vec<f64> = edges
            .iter()
            .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
            .collect();
        let graph = IsingGraph::new(vec![0.0; n], j, edges);
        let int = IntGraph::from_base(&graph).unwrap();
        let cpu = CpuGraph::from_base(&graph);
        let counts = bond_counts(&int).unwrap();
        let betas = geometric_beta_schedule(0.017, 6.5, 550);
        let table = acceptance_table(&betas, int.row_len());
        let mut seed_rng = SmallRng::seed_from_u64(99);
        let draws = threshold_draws(&betas, int.max_field(), &mut seed_rng);
        let offsets = sweep_offsets(betas.len(), 1, &mut SmallRng::seed_from_u64(5));
        let reads = 4u64;
        let updates = (reads as f64) * (betas.len() as f64) * (n as f64);

        let mut times = Vec::new();
        for which in 0..3 {
            let t = Instant::now();
            for r in 0..reads {
                let mut g = SmallRng::seed_from_u64(r);
                match which {
                    0 => {
                        anneal_one_read(&cpu, &betas, 1, &mut g, None).unwrap();
                    }
                    1 => {
                        anneal_one_read_int(&int, &table, 1, &mut g, None).unwrap();
                    }
                    _ => {
                        let mut spins = random_spins(n, &mut g);
                        anneal_from(&int, &draws, 1, &mut spins, &offsets, None).unwrap();
                    }
                }
            }
            times.push(t.elapsed().as_secs_f64());
        }

        let t = Instant::now();
        let mut g = SmallRng::seed_from_u64(1);
        let mut state = MscState::random(n, &mut g);
        anneal_word(&int, &counts, &draws, 1, &mut state, &offsets, None).unwrap();
        let word = t.elapsed().as_secs_f64();
        let word_updates = LANES as f64 * betas.len() as f64 * n as f64;

        println!(
            "{name:7} n={n} | f64 {:.0} M/s | int {:.0} M/s ({:.2}x) | fast {:.0} M/s ({:.2}x) \
             | msc {:.0} M/s ({:.1}x raw, {:.1}x per 36-read job)",
            updates / times[0] / 1e6,
            updates / times[1] / 1e6,
            times[0] / times[1],
            updates / times[2] / 1e6,
            times[0] / times[2],
            word_updates / word / 1e6,
            (word_updates / word) / (updates / times[0]),
            (times[0] / reads as f64 * 36.0) / word,
        );
    }
}
