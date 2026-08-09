//! SB particle-level parallelism: workers split the particles inside one read.
use quip_miner_core::{IsingGraph, SampleParams};
use quip_miner_cpu::{sample_sb_with_workers, DSB};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn pivot(scale: usize) -> IsingGraph {
    let raw = std::fs::read_to_string(
        "/Users/carback1/Code/quip/isingmark/fixtures/advantage2-system1.spec.json").unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let ids: Vec<usize> = v["nodes"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    let mut idx = vec![usize::MAX; ids.iter().copied().max().unwrap()+1];
    for (d,&i) in ids.iter().enumerate() { idx[i]=d; }
    let base: Vec<(usize,usize)> = v["edges"].as_array().unwrap().iter()
        .map(|e| (idx[e[0].as_u64().unwrap() as usize], idx[e[1].as_u64().unwrap() as usize])).collect();
    let n = ids.len();
    let mut edges = Vec::new();
    for t in 0..scale { for &(a,b) in &base { edges.push((a+t*n, b+t*n)); } }
    let mut rng = SmallRng::seed_from_u64(900);
    let j: Vec<f64> = (0..edges.len()).map(|_| if rng.gen::<bool>() {1.0} else {-1.0}).collect();
    IsingGraph::new(vec![0.0; n*scale], j, edges)
}

fn main() {
    for (scale, label) in [(1usize,"4577"), (4,"18308")] {
        let g = pivot(scale);
        // One read, so the only parallelism available is across particles.
        let p = SampleParams { num_reads: 1, num_sweeps: 1000, seed: 42, ..Default::default() };
        println!("\n## {label} nodes, {} edges, 1 read x 1000 steps", g.edges.len());
        println!("| workers | median | speedup | best energy |");
        println!("|--------:|-------:|--------:|------------:|");
        let mut base = 0.0f64;
        for w in [1usize,2,4,6,8,12,16] {
            let mut ts=Vec::new(); let mut best=0i64;
            for _ in 0..5 {
                let t0=Instant::now();
                let r = sample_sb_with_workers(&g,&p,DSB,w);
                ts.push(t0.elapsed().as_secs_f64()*1e3);
                best = r.iter().map(|x| x.energy_milli).min().unwrap();
            }
            ts.sort_by(|a,b| a.partial_cmp(b).unwrap());
            let s=ts[2];
            if w==1 { base=s; }
            println!("| {w} | {s:.0} ms | {:.2}x | {best} |", base/s);
        }
    }
}
