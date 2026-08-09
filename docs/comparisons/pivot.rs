use quip_miner_core::{Algorithm, IsingGraph, SampleParams};
use quip_miner_cpu::gibbs_parallel::{sample_gibbs_parallel, DEFAULT_GIBBS_WORKERS};
use quip_miner_cpu::{sample_ising, sample_sb, DSB};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn pivot() -> IsingGraph {
    let raw = std::fs::read_to_string(
        "/Users/carback1/Code/quip/isingmark/fixtures/advantage2-system1.spec.json").unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let ids: Vec<usize> = v["nodes"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    let mut idx = vec![usize::MAX; ids.iter().copied().max().unwrap()+1];
    for (d,&i) in ids.iter().enumerate() { idx[i]=d; }
    let edges: Vec<(usize,usize)> = v["edges"].as_array().unwrap().iter()
        .map(|e| (idx[e[0].as_u64().unwrap() as usize], idx[e[1].as_u64().unwrap() as usize])).collect();
    let mut rng = SmallRng::seed_from_u64(900);
    let j: Vec<f64> = (0..edges.len()).map(|_| if rng.gen::<bool>() {1.0} else {-1.0}).collect();
    IsingGraph::new(vec![0.0; ids.len()], j, edges)
}

fn main() {
    let g = pivot();
    let p = SampleParams { num_reads: 16, num_sweeps: 1000, seed: 42, ..Default::default() };
    println!("| kernel | cores | median | min | max | best energy |");
    println!("|---|---:|---:|---:|---:|---:|");
    for name in ["cpu-sa","cpu-gibbs","cpu-sb"] {
        let mut ts=Vec::new(); let mut best=0i64;
        for _ in 0..7 {
            let t0=Instant::now();
            let r = match name {
                "cpu-sa" => sample_ising(&g,&p,Algorithm::Sa),
                "cpu-gibbs" => sample_gibbs_parallel(&g,&p,DEFAULT_GIBBS_WORKERS),
                _ => sample_sb(&g,&p,DSB),
            };
            ts.push(t0.elapsed().as_secs_f64()*1e3);
            best = r.iter().map(|x| x.energy_milli).min().unwrap();
        }
        ts.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let cores = if name=="cpu-gibbs" { DEFAULT_GIBBS_WORKERS } else { 1 };
        println!("| {name} | {cores} | {:.0} ms | {:.0} | {:.0} | {best} |", ts[3], ts[0], ts[6]);
    }
}
