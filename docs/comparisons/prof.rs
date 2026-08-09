use quip_miner_core::{Algorithm, IsingGraph, SampleParams};
use quip_miner_cpu::{sample_ising, sample_sb, DSB};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn main() {
    let which = std::env::args().nth(1).expect("kernel");
    let n = 4000usize;
    let mut rng = SmallRng::seed_from_u64(1);
    let mut edges = Vec::new();
    for _ in 0..36000 {
        let u = rng.gen_range(0..n);
        let v = rng.gen_range(0..n);
        if u != v { edges.push((u.min(v), u.max(v))); }
    }
    let j: Vec<f64> = (0..edges.len()).map(|_| if rng.gen::<bool>() {1.0} else {-1.0}).collect();
    let g = IsingGraph::new(vec![0.0; n], j, edges);
    let p = SampleParams { num_reads: 16, num_sweeps: 1000, seed: 7, ..Default::default() };
    let out = match which.as_str() {
        "sa" => sample_ising(&g, &p, Algorithm::Sa),
        "gibbs" => sample_ising(&g, &p, Algorithm::Gibbs),
        _ => sample_sb(&g, &p, DSB),
    };
    eprintln!("{} best {}", which, out.iter().map(|r| r.energy_milli).min().unwrap());
}
