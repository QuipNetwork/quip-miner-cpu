//! End-to-end: a synthetic bench run writes a valid report whose top-level
//! parts sum to the measured model time within a small residual.

use quip_miner_cpu::bench::report::{BenchReport, TOP_LEVEL_PARTS};
use quip_miner_cpu::bench::{run_bench, BenchArgs, SourceKind};
use quip_miner_cpu::Algorithm;
use std::path::PathBuf;

fn out_dir() -> PathBuf {
    let d = std::env::temp_dir().join(format!("quipbench-e2e-{}", std::process::id()));
    std::fs::create_dir_all(&d).expect("mkdir");
    d
}

#[test]
fn synthetic_bench_report_is_self_consistent() {
    let dir = out_dir();
    let args = BenchArgs {
        algorithm: Algorithm::Sa,
        source: SourceKind::Synthetic {
            n_nodes: 64,
            n_edges: 256,
        },
        num_reads: 8,
        num_sweeps: 256,
        sweeps_per_beta: 1,
        seed: 7,
        warmup: 2,
        iters: 5,
        out_dir: dir.clone(),
    };
    run_bench(&args).expect("bench run");

    // Exactly one report written; load and validate it.
    let json_path = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.extension().map(|x| x == "json").unwrap_or(false))
        .expect("a .json report");
    let rep: BenchReport =
        serde_json::from_str(&std::fs::read_to_string(&json_path).unwrap()).expect("parse report");

    assert_eq!(rep.schema_version, 1);
    assert_eq!(rep.backend, "cpu");
    assert_eq!(rep.algorithm, "sa");
    assert_eq!(rep.model.n_nodes, 64);
    // spin_visits = iters(5) * reads(8) * betas(256) * sweeps_per_beta(1) * n(64):
    // the aggregator sums per-part busy-time across all measured iterations, so
    // the visit count must scale with iters too.
    assert_eq!(rep.derived.spin_visits, 5 * 8 * 256 * 64);
    assert!(rep.derived.per_spin_ns > 0.0);
    assert!((0.0..=1.0).contains(&rep.derived.accept_rate));

    // Self-consistency: summed top-level parts within 15% of measured time.
    let summed: u64 = rep
        .parts
        .iter()
        .filter(|p| TOP_LEVEL_PARTS.contains(&p.part.as_str()))
        .map(|p| p.total_ns)
        .sum();
    assert!(summed > 0);
    assert!(
        rep.residual_frac.abs() < 0.15,
        "residual_frac {} too large (measured {} vs summed {})",
        rep.residual_frac,
        rep.measured_model_ns,
        summed
    );

    // Folded stacks written and non-empty.
    let folded = json_path.with_extension("folded");
    assert!(std::fs::metadata(&folded)
        .map(|m| m.len() > 0)
        .unwrap_or(false));

    let _ = std::fs::remove_dir_all(&dir);
}
