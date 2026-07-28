//! End-to-end: a tiny 4-node corpus + `topology.spec.json` redraws a nonce
//! entry into a real model and benches it, producing a valid per-part report.
//!
//! This is the coordinator's actual on-disk shape: `instances.jsonl` lines
//! keyed on `nonce` with unrelated keys riding along, and topology supplied
//! separately via `--topology <spec.json>` (`{nodes, edges, allowed_h_milli,
//! allowed_j_milli}`) rather than embedded per-line.

use std::io::Write as _;
use std::path::PathBuf;

use quip_miner_cpu::bench::report::BenchReport;
use quip_miner_cpu::bench::{run_bench, BenchArgs, SourceKind};
use quip_miner_cpu::Algorithm;

fn scratch_dir() -> PathBuf {
    let d = std::env::temp_dir().join(format!("quipcorpusbench-{}", std::process::id()));
    std::fs::create_dir_all(&d).expect("mkdir");
    d
}

#[test]
fn nonce_entry_from_real_corpus_shape_redraws_and_benches() {
    let dir = scratch_dir();

    // Coordinator's per-bucket topology.spec.json.
    let topo_path = dir.join("topology.spec.json");
    std::fs::write(
        &topo_path,
        r#"{
            "nodes": [0, 1, 2, 3],
            "edges": [[0, 1], [1, 2], [2, 3], [0, 3]],
            "allowed_h_milli": [-1000, 0, 1000],
            "allowed_j_milli": [-1000, 1000]
        }"#,
    )
    .expect("write topology spec");

    // Coordinator's instances.jsonl: nonce-keyed, extra keys ride along.
    let corpus_path = dir.join("instances.jsonl");
    let mut f = std::fs::File::create(&corpus_path).expect("create corpus");
    writeln!(
        f,
        r#"{{"nonce":"{}","topology_hash":"deadbeef","energy_milli":-1500,"salt_hex":"aa","qblock_id":7,"difficulty":{{"max_energy_milli":-1000,"min_solutions":1,"min_diversity_milli":0}}}}"#,
        "ab".repeat(32)
    )
    .expect("write corpus line");
    drop(f);

    let out_dir = dir.join("bench-out");
    let args = BenchArgs {
        algorithm: Algorithm::Sa,
        source: SourceKind::Corpus {
            path: corpus_path,
            topology: Some(topo_path),
        },
        num_reads: 4,
        num_sweeps: 8,
        sweeps_per_beta: 1,
        seed: 1,
        warmup: 0,
        iters: 1,
        out_dir: out_dir.clone(),
        limit: None,
    };
    run_bench(&args).expect("run_bench over nonce-ref corpus");

    let json_path = std::fs::read_dir(&out_dir)
        .expect("read out dir")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|ext| ext == "json"))
        .expect("one json report written");
    let rep: BenchReport =
        serde_json::from_str(&std::fs::read_to_string(&json_path).unwrap()).expect("parse report");

    assert_eq!(rep.schema_version, 1);
    assert_eq!(rep.backend, "cpu");
    assert_eq!(rep.model.n_nodes, 4);
    assert_eq!(rep.model.n_edges, 4);
    assert!(!rep.parts.is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}
