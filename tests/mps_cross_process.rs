//! Cross-process reproducibility for the tensor-network kernel.
//!
//! Two in-process calls are not enough here. The kernel used to cut its anneal
//! short on a wall-clock deadline, so its output depended on how fast the host
//! got through the loop. Two calls inside one process meet the same cache state
//! and the same load and agreed often enough to look fine, while two separate
//! invocations disagreed. Anything else that varies per process rather than per
//! call, address-space layout or an environment-seeded hasher among them, hides
//! from an in-process check the same way.
//!
//! So this spawns the test binary again, twice, and compares what each child
//! writes. The child re-runs one named test in this file, which returns early
//! unless `MPS_CHILD_OUT` names a file to write.
//!
//! A reproducibility check is only worth as much as its fixture's sensitivity.
//! `a_one_step_shorter_anneal_changes_the_records` is the control: it shows this
//! fixture notices a single Trotter step, so the cross-process check would catch
//! anything that lets the machine decide where the anneal stops.

use quip_miner_cpu::{sample_ising_mps, InitMode, IsingGraph, MpsConfig, SampleParams};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Names the output file for a child run. Absent in the parent.
const CHILD_OUT: &str = "MPS_CHILD_OUT";

/// Work budget for a child run, so the parent can ask for a shorter anneal.
const CHILD_BUDGET: &str = "MPS_CHILD_BUDGET";

/// The test the child re-runs. Kept beside the function it names so a rename
/// that misses one of them fails loudly rather than quietly running no test.
const CHILD_TEST: &str = "child_writes_records";

/// Trims the fixture's anneal to 77 Trotter steps: `1.6e8 / (50 * 1531 * 3^3)`
/// is 77.4, and `select_steps` floors it. Keeps the run cheap.
const STEP_BUDGET: f64 = 1.6e8;

/// Buys 76 steps on the same fixture: `1.58e8 / (50 * 1531 * 3^3)` is 76.5.
const STEP_BUDGET_ONE_LESS: f64 = 1.58e8;

/// A banded chain of 512 sites. The span sum is `511 + 2 * 510` = 1531, small
/// enough that `select_chi` picks 3 rather than flooring at 1. Bond dimension
/// matters: a mean-field state polarizes early, so at chi 1 a step either side
/// samples the same configuration and the fixture would notice nothing. Built
/// from a fixed pattern with no RNG, so it cannot drift.
fn banded_chain() -> IsingGraph {
    let (n, band) = (512usize, 2usize);
    let mut edges = Vec::new();
    let mut j = Vec::new();
    for u in 0..n {
        for d in 1..=band {
            if u + d < n {
                edges.push((u, u + d));
                // Mixed signs: frustrated enough that the reads do not all
                // collapse onto one configuration.
                j.push(if (u + d) % 3 == 0 { 1.0 } else { -1.0 });
            }
        }
    }
    IsingGraph::new(vec![0.0; n], j, edges)
}

/// The fixture both checks share. `time_budget_ms` is 0, matching what
/// `MpsConfig::from_env` ships, so nothing here reads the clock.
fn fixture(budget: f64) -> (MpsConfig, SampleParams, IsingGraph) {
    let params = SampleParams {
        num_reads: 4,
        num_sweeps: 320,
        seed: 4_242,
        ..Default::default()
    };
    let cfg = MpsConfig {
        chi_max: 32,
        init: InitMode::Anneal,
        time_budget_ms: 0,
        flop_budget: 1.25e9,
        anneal_work_budget: budget,
    };
    (cfg, params, banded_chain())
}

fn records(budget: f64) -> String {
    let (cfg, params, graph) = fixture(budget);
    let mut out = String::new();
    for r in sample_ising_mps(&graph, &params, &cfg) {
        out.push_str(&r.energy_milli.to_string());
        out.push(' ');
        for s in &r.spins {
            out.push(if *s == 1 { '+' } else { '-' });
        }
        out.push('\n');
    }
    out
}

/// Child entry point. A normal run reaches this with `MPS_CHILD_OUT` unset and
/// does nothing, so it costs the suite one no-op.
#[test]
fn child_writes_records() {
    let Ok(path) = std::env::var(CHILD_OUT) else {
        return;
    };
    let budget = std::env::var(CHILD_BUDGET)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(STEP_BUDGET);
    std::fs::write(path, records(budget)).expect("child writes its records");
}

fn run_child(exe: &Path, out: &Path, budget: f64) -> String {
    let status = Command::new(exe)
        .args(["--exact", CHILD_TEST, "--quiet"])
        .env(CHILD_OUT, out)
        .env(CHILD_BUDGET, budget.to_string())
        // The anneal arm is the one under test; do not inherit a stray value.
        .env_remove("QUIP_MPS_INIT")
        .status()
        .expect("spawn the test binary again");
    assert!(status.success(), "child run failed: {status}");
    std::fs::read_to_string(out).expect("child wrote no records")
}

fn temp_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("quip-mps-xproc-{}-{tag}", std::process::id()));
    p
}

#[test]
fn the_anneal_is_reproducible_across_processes() {
    let exe = std::env::current_exe().expect("path to this test binary");
    let a_path = temp_path("a");
    let b_path = temp_path("b");

    let a = run_child(&exe, &a_path, STEP_BUDGET);
    let b = run_child(&exe, &b_path, STEP_BUDGET);

    let _ = std::fs::remove_file(&a_path);
    let _ = std::fs::remove_file(&b_path);

    assert!(!a.is_empty(), "the child produced no records");
    assert_eq!(
        a, b,
        "two separate processes gave different records for one seed, so \
         something outside the seed is steering the kernel"
    );
}

/// The control for the check above. A fixture that shrugged off a step would
/// pass that check whatever the anneal did.
#[test]
fn a_one_step_shorter_anneal_changes_the_records() {
    let full = records(STEP_BUDGET);
    let short = records(STEP_BUDGET_ONE_LESS);
    assert_ne!(
        full, short,
        "this fixture cannot tell a 77-step anneal from a 76-step one, so it \
         cannot detect a clock deciding where the anneal stops"
    );
}

/// The shipped configuration must take no input from the clock. This is the
/// property that makes the kernel reproducible; the cross-process check above
/// would not notice a wall-clock cut-off that never fired on the test host.
#[test]
fn the_shipped_configuration_reads_no_clock() {
    assert_eq!(
        MpsConfig::from_env(32).time_budget_ms,
        0,
        "a non-zero wall-clock budget makes the same seed give different \
         answers on a different machine, under load, or in a debug build"
    );
}
