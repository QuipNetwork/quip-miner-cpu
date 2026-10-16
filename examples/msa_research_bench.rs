// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Scalar SA against unit and float MSA on identical models, at matched sweeps
//! and at matched wall time. Run on an idle host:
//!
//! ```sh
//! cargo run --release --example msa_research_bench -- --models 5 --repeats 3
//! ```
//!
//! Prints one JSON object per (regime, kernel, sweeps, model) on stdout.

use std::collections::BTreeSet;
use std::io::{self, Write};
use std::time::Instant;

use clap::Parser;
use quip_solver_cpu::research::{sample_research, ResearchKernel};
use quip_solver_cpu::{IsingGraph, SampleParams};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

#[derive(Parser)]
struct Args {
    /// Models per regime.
    #[arg(long, default_value_t = 5)]
    models: u64,
    /// Timed repeats per point; the median is reported.
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    /// Reads per call.
    #[arg(long, default_value_t = 64)]
    reads: usize,
}

const SWEEPS: [usize; 3] = [512, 2048, 8192];

fn ring(
    n: usize,
    degree: usize,
    seed: u64,
    coupling: impl Fn(&mut SmallRng) -> f64,
    field: impl Fn(&mut SmallRng) -> f64,
) -> IsingGraph {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut edges = Vec::new();
    for v in 0..n {
        for k in 1..=degree / 2 {
            edges.push((v, (v + k) % n));
        }
    }
    let j = edges.iter().map(|_| coupling(&mut rng)).collect();
    let h = (0..n).map(|_| field(&mut rng)).collect();
    IsingGraph::new(h, j, edges)
}

fn cubic(side: usize, seed: u64) -> IsingGraph {
    let mut rng = SmallRng::seed_from_u64(seed);
    let at = |x: usize, y: usize, z: usize| (x % side) + side * ((y % side) + side * (z % side));
    let mut edges = Vec::new();
    for x in 0..side {
        for y in 0..side {
            for z in 0..side {
                let v = at(x, y, z);
                edges.push((v, at(x + 1, y, z)));
                edges.push((v, at(x, y + 1, z)));
                edges.push((v, at(x, y, z + 1)));
            }
        }
    }
    let j = edges
        .iter()
        .map(|_| if rng.gen() { 1.0 } else { -1.0 })
        .collect();
    IsingGraph::new(vec![0.0; side * side * side], j, edges)
}

fn clique(n: usize, seed: u64) -> IsingGraph {
    let mut rng = SmallRng::seed_from_u64(seed);
    let edges: Vec<(usize, usize)> = (0..n)
        .flat_map(|u| ((u + 1)..n).map(move |v| (u, v)))
        .collect();
    let j = edges.iter().map(|_| rng.gen_range(-1.0..1.0)).collect();
    let h = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    IsingGraph::new(h, j, edges)
}

fn signed(rng: &mut SmallRng, lo: f64, hi: f64) -> f64 {
    let v = rng.gen_range(lo..=hi);
    if rng.gen() {
        v
    } else {
        -v
    }
}

fn regimes(seed: u64) -> Vec<(&'static str, IsingGraph, Vec<ResearchKernel>)> {
    use ResearchKernel::{FloatMsa, ScalarSa, UnitMsa};
    vec![
        (
            "native-pm1",
            ring(
                4096,
                20,
                seed,
                |r| if r.gen() { 1.0 } else { -1.0 },
                |_| 0.0,
            ),
            vec![ScalarSa, UnitMsa, FloatMsa],
        ),
        (
            "fractional",
            ring(
                4096,
                20,
                seed,
                |r| signed(r, 0.01, 1.0),
                |r| signed(r, 0.01, 1.0),
            ),
            vec![ScalarSa, FloatMsa],
        ),
        (
            "multilevel",
            ring(
                4096,
                20,
                seed,
                |r| f64::from(r.gen_range(1..=7)) * if r.gen() { 1.0 } else { -1.0 },
                |r| f64::from(r.gen_range(-3..=3)),
            ),
            vec![ScalarSa, FloatMsa],
        ),
        (
            "clique-portfolio",
            clique(108, seed),
            vec![ScalarSa, FloatMsa],
        ),
        (
            "cubic-pm1",
            cubic(13, seed),
            vec![ScalarSa, UnitMsa, FloatMsa],
        ),
    ]
}

struct Point {
    seconds: f64,
    best: f64,
    mean: f64,
    distinct: usize,
    workspace: Option<usize>,
}

fn run(
    graph: &IsingGraph,
    kernel: ResearchKernel,
    sweeps: usize,
    reads: usize,
    seed: u64,
    repeats: usize,
) -> Point {
    let params = SampleParams {
        num_reads: reads,
        num_sweeps: sweeps,
        seed,
        ..Default::default()
    };
    let mut times = Vec::with_capacity(repeats);
    let mut last = None;
    for _ in 0..repeats {
        let start = Instant::now();
        let out = sample_research(graph, &params, kernel, None, None)
            .expect("benchmark models are valid");
        times.push(start.elapsed().as_secs_f64());
        last = Some(out);
    }
    times.sort_by(f64::total_cmp);
    let out = last.expect("repeats >= 1");
    let best = out.energies.iter().copied().fold(f64::INFINITY, f64::min);
    let mean = out.energies.iter().sum::<f64>() / out.energies.len() as f64;
    let distinct = out.spins.iter().collect::<BTreeSet<_>>().len();
    Point {
        seconds: times[times.len() / 2],
        best,
        mean,
        distinct,
        workspace: out.metadata.workspace_bytes,
    }
}

fn emit(
    regime: &str,
    kernel: ResearchKernel,
    budget: &str,
    sweeps: usize,
    model: u64,
    p: &Point,
) -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    writeln!(
        stdout,
        "{{\"regime\":\"{regime}\",\"kernel\":\"{}\",\"budget\":\"{budget}\",\"sweeps\":{sweeps},\"model\":{model},\"seconds\":{:.6},\"best\":{:.6},\"mean\":{:.6},\"distinct\":{},\"workspace_bytes\":{}}}",
        kernel.name(),
        p.seconds,
        p.best,
        p.mean,
        p.distinct,
        p.workspace.map_or("null".to_owned(), |w| w.to_string()),
    )
}

fn main() -> io::Result<()> {
    match run_bench() {
        Err(error) if error.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        result => result,
    }
}

fn run_bench() -> io::Result<()> {
    let args = Args::parse();
    if args.repeats == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--repeats must be at least 1",
        ));
    }
    for model in 0..args.models {
        for (regime, graph, kernels) in regimes(model) {
            for &sweeps in &SWEEPS {
                let mut float_seconds = None;
                for &kernel in &kernels {
                    let p = run(&graph, kernel, sweeps, args.reads, model, args.repeats);
                    if kernel == ResearchKernel::FloatMsa {
                        float_seconds = Some(p.seconds);
                    }
                    emit(regime, kernel, "matched-sweeps", sweeps, model, &p)?;
                }
                // Matched time: scalar SA at the sweep count that takes as long
                // as the float kernel took, from scalar SA's measured rate.
                if let Some(target) = float_seconds {
                    let probe = run(
                        &graph,
                        ResearchKernel::ScalarSa,
                        sweeps,
                        args.reads,
                        model,
                        1,
                    );
                    let scaled =
                        ((sweeps as f64) * target / probe.seconds).round().max(1.0) as usize;
                    let p = run(
                        &graph,
                        ResearchKernel::ScalarSa,
                        scaled,
                        args.reads,
                        model,
                        args.repeats,
                    );
                    emit(
                        regime,
                        ResearchKernel::ScalarSa,
                        "matched-time",
                        scaled,
                        model,
                        &p,
                    )?;
                }
            }
        }
    }
    Ok(())
}
