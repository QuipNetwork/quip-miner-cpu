//! Gibbs read-level parallelism: whole reads to whole threads, one worker each,
//! so the class barrier never runs. Compare against splitting classes.
use quip_miner_core::{IsingGraph, SampleParams};
use quip_miner_cpu::gibbs_parallel::sample_gibbs_parallel;
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
    let total = 16usize;
    println!("| threads | reads each | min | median | max | speedup on min |");
    println!("|--------:|-----------:|----:|-------:|----:|---------------:|");
    let mut base = 0.0f64;
    for t in [1usize,2,4,8,16] {
        let per = total / t;
        let mut ts = Vec::new();
        for _ in 0..9 {
            let t0 = Instant::now();
            std::thread::scope(|s| {
                for k in 0..t {
                    let g = &g;
                    s.spawn(move || {
                        let p = SampleParams { num_reads: per, num_sweeps: 1000,
                                               seed: 42 + k as u64, ..Default::default() };
                        // One worker per read: no class barrier at all.
                        sample_gibbs_parallel(g, &p, 1)
                    });
                }
            });
            ts.push(t0.elapsed().as_secs_f64());
        }
        ts.sort_by(|a,b| a.partial_cmp(b).unwrap());
        if t==1 { base = ts[0]; }
        println!("| {t} | {per} | {:.3} s | {:.3} s | {:.3} s | {:.2}x |", ts[0], ts[4], ts[8], base/ts[0]);
    }
}
