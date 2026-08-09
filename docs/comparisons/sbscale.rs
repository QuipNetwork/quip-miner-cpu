//! SB read-level parallelism: reads are independent trajectories, so splitting
//! 16 reads across T threads needs no kernel change and no synchronisation.
use quip_miner_core::{IsingGraph, SampleParams};
use quip_miner_cpu::{sample_sb, DSB};
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
    let total_reads = 16usize;
    println!("| threads | reads/thread | median wall | speedup |");
    println!("|--------:|-------------:|------------:|--------:|");
    let mut base = 0.0f64;
    for t in [1usize,2,4,6,8,12,16] {
        if total_reads % t != 0 { continue; }
        let per = total_reads / t;
        let mut ts = Vec::new();
        for _ in 0..5 {
            let t0 = Instant::now();
            std::thread::scope(|s| {
                for k in 0..t {
                    let g = &g;
                    s.spawn(move || {
                        let p = SampleParams { num_reads: per, num_sweeps: 1000,
                                               seed: 42 + k as u64, ..Default::default() };
                        sample_sb(g, &p, DSB)
                    });
                }
            });
            ts.push(t0.elapsed().as_secs_f64());
        }
        ts.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let s = ts[2];
        if t==1 { base = s; }
        println!("| {t} | {per} | {s:.2} s | {:.2}x |", base/s);
    }
}
