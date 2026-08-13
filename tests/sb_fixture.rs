//! Local determinism regression fixture for the SB kernel.
//!
//! This is NOT a consensus artifact. `conformance/golden_vectors.json` pins
//! `energy_milli` against the Python reference. This file pins only that the SB
//! kernel keeps producing the same spins for the same input on this build. A
//! deliberate change to DT, A0, INIT_RANGE, the integration order, or the
//! per-read seed derivation is expected to change it, and is not a protocol
//! break.
//!
//! Regenerate and review the diff with:
//!
//! ```sh
//! UPDATE_FIXTURE=1 cargo test --release -p quip-miner-cpu \
//!   --test sb_fixture sb_kernel_matches_the_determinism_fixture -- --exact
//! ```

use quip_miner_cpu::{sample_sb, IsingGraph, SampleParams, SbVariant, BSB, DSB, HBSB, HDSB};
use serde_json::{json, Value};
use std::fs;

const FIXTURE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/conformance/sb_determinism.json"
);

struct Case {
    name: &'static str,
    algorithm: &'static str,
    variant: SbVariant,
    graph: IsingGraph,
    params: SampleParams,
}

fn params(num_reads: usize, num_sweeps: usize, seed: u64) -> SampleParams {
    SampleParams {
        num_reads,
        num_sweeps,
        seed,
        ..Default::default()
    }
}

fn ferro_pair() -> IsingGraph {
    IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)])
}

fn biased_chain() -> IsingGraph {
    IsingGraph::new(vec![2.0, -0.5, 0.25], vec![-1.0, 1.0], vec![(0, 1), (1, 2)])
}

fn ring8() -> IsingGraph {
    let edges: Vec<(usize, usize)> = (0..8).map(|i| (i, (i + 1) % 8)).collect();
    let j: Vec<f64> = (0..8)
        .map(|k| if k % 2 == 0 { -1.0 } else { 1.0 })
        .collect();
    IsingGraph::new(vec![0.0; 8], j, edges)
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "ferro_pair",
            algorithm: "sb",
            variant: DSB,
            graph: ferro_pair(),
            params: params(4, 64, 1),
        },
        Case {
            name: "biased_chain",
            algorithm: "sb",
            variant: DSB,
            graph: biased_chain(),
            params: params(4, 128, 2),
        },
        Case {
            name: "ring8_ballistic",
            algorithm: "bsb",
            variant: BSB,
            graph: ring8(),
            params: params(4, 128, 3),
        },
        Case {
            name: "ring8_heated_discrete",
            algorithm: "hdsb",
            variant: HDSB,
            graph: ring8(),
            params: params(4, 128, 4),
        },
        Case {
            name: "ring8_heated_ballistic",
            algorithm: "hbsb",
            variant: HBSB,
            graph: ring8(),
            params: params(4, 128, 5),
        },
    ]
}

fn observed() -> Value {
    let entries: Vec<Value> = cases()
        .into_iter()
        .map(|c| {
            let results = sample_sb(&c.graph, &c.params, c.variant);
            json!({
                "name": c.name,
                "algorithm": c.algorithm,
                "h": c.graph.h,
                "j": c.graph.j,
                "edges": c.graph.edges,
                "num_reads": c.params.num_reads,
                "num_sweeps": c.params.num_sweeps,
                "seed": c.params.seed,
                "spins": results.iter().map(|r| r.spins.clone()).collect::<Vec<_>>(),
                "energy_milli": results.iter().map(|r| r.energy_milli).collect::<Vec<_>>(),
            })
        })
        .collect();
    json!({
        "_comment": "Local regression fixture for the SB kernel. Not a consensus \
    artifact. Regenerate with UPDATE_FIXTURE=1 when a kernel constant changes.",
        "cases": entries,
    })
}

/// Hypothesis: the SB kernel is stable across refactoring. A change here means
/// either a deliberate constant change, which is regenerated and reviewed, or an
/// accidental behavior change, which is a bug.
#[test]
fn sb_kernel_matches_the_determinism_fixture() {
    let observed = observed();
    if std::env::var("UPDATE_FIXTURE").as_deref() == Ok("1") {
        let text = serde_json::to_string_pretty(&observed).expect("serialize fixture");
        fs::write(FIXTURE_PATH, format!("{text}\n")).expect("write fixture");
        return;
    }
    let raw = fs::read_to_string(FIXTURE_PATH).expect("read conformance/sb_determinism.json");
    let expected: Value =
        serde_json::from_str(&raw).expect("parse conformance/sb_determinism.json");
    assert_eq!(
        observed, expected,
        "SB kernel output drifted from conformance/sb_determinism.json. If the change \
         is deliberate, regenerate with UPDATE_FIXTURE=1 and review the diff."
    );
}
