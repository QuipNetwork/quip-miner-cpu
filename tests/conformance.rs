//! Protocol conformance: spawn SA and Gibbs miners against quip-solver-conformance.

use quip_proto::v1::RejectReason;
use quip_solver_conformance::driver::{drive_miner, DriverReport};
use quip_solver_conformance::solve::{drive_solve, CaseOutcome};
use std::process::Command;
use std::sync::LazyLock;
use tokio::sync::Mutex;

/// Serializes this file's tests. Each spawns a real miner process, and
/// `cargo test` runs `#[tokio::test]` cases concurrently on separate threads
/// by default; a miner process (e.g. `quip-cpu-gibbs`) also allocates its own
/// worker threads on top of that, so unbounded parallelism can starve one
/// past the mock coordinator's job deadline and drop a result. These are
/// process-spawning integration tests, so serializing them costs little
/// wall-clock time and buys back determinism. See qrel-p02. The guard is held
/// across the drive await, so the mutex must be the async-aware one.
static GATE: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

/// Drives a miner process while holding [`GATE`], so at most one test in
/// this file spawns a miner at a time.
async fn drive_miner_bounded(bin_path: &str, socket: &str) -> DriverReport {
    let _guard = GATE.lock().await;
    drive_miner(bin_path, socket).await
}

/// Grade the binary built by Cargo against the one-shot solve contract, and
/// check the linear-schedule case ran against the right expectation: solved
/// for an identity that lists `linear-schedule`, refused for one that does
/// not. The gate grades both, so this pins which one each binary claims.
async fn assert_solve_conformance(bin: &str) {
    let _guard = GATE.lock().await;
    let report = drive_solve(&profile_bin(bin)).await;
    assert!(
        report.is_conformant(),
        "{bin} --solve failed conformance:\n{}",
        report.summary()
    );
    assert_eq!(
        report.outcome("linear-schedule"),
        Some(&CaseOutcome::Passed),
        "{bin}:\n{}",
        report.summary()
    );
}

#[tokio::test]
async fn quip_cpu_sa_passes_solve_conformance() {
    assert_solve_conformance("quip-cpu-sa").await;
}

#[tokio::test]
async fn quip_cpu_gibbs_passes_solve_conformance() {
    assert_solve_conformance("quip-cpu-gibbs").await;
}

#[tokio::test]
async fn quip_cpu_sb_passes_solve_conformance() {
    assert_solve_conformance("quip-cpu-sb").await;
}

#[cfg(feature = "experimental")]
#[tokio::test]
async fn quip_cpu_bsb_passes_solve_conformance() {
    assert_solve_conformance("quip-cpu-bsb").await;
}

#[cfg(feature = "experimental")]
#[tokio::test]
async fn quip_cpu_hdsb_passes_solve_conformance() {
    assert_solve_conformance("quip-cpu-hdsb").await;
}

#[cfg(feature = "experimental")]
#[tokio::test]
async fn quip_cpu_hbsb_passes_solve_conformance() {
    assert_solve_conformance("quip-cpu-hbsb").await;
}

#[cfg(feature = "experimental")]
#[tokio::test]
async fn quip_cpu_gbsb_passes_solve_conformance() {
    assert_solve_conformance("quip-cpu-gbsb").await;
}

#[cfg(feature = "experimental")]
#[tokio::test]
async fn quip_cpu_gdsb_passes_solve_conformance() {
    assert_solve_conformance("quip-cpu-gdsb").await;
}

#[cfg(feature = "experimental")]
#[tokio::test]
async fn quip_cpu_tedsb_passes_solve_conformance() {
    assert_solve_conformance("quip-cpu-tedsb").await;
}

#[cfg(feature = "experimental")]
#[tokio::test]
async fn quip_cpu_sbqa_passes_solve_conformance() {
    assert_solve_conformance("quip-cpu-sbqa").await;
}

#[cfg(feature = "experimental")]
#[tokio::test]
async fn quip_cpu_ggdsb_passes_solve_conformance() {
    assert_solve_conformance("quip-cpu-ggdsb").await;
}

#[cfg(feature = "experimental")]
#[tokio::test]
async fn quip_cpu_fsa_passes_solve_conformance() {
    assert_solve_conformance("quip-cpu-fsa").await;
}

#[cfg(feature = "experimental")]
#[tokio::test]
async fn quip_cpu_msa_passes_solve_conformance() {
    assert_solve_conformance("quip-cpu-msa").await;
}

#[cfg(feature = "experimental")]
#[tokio::test]
async fn quip_cpu_mps_passes_solve_conformance() {
    assert_solve_conformance("quip-cpu-mps").await;
}

#[cfg(feature = "experimental")]
#[tokio::test]
async fn quip_cpu_mfa_passes_solve_conformance() {
    assert_solve_conformance("quip-cpu-mfa").await;
}

#[cfg(feature = "experimental")]
#[tokio::test]
async fn quip_cpu_flatiron_passes_solve_conformance() {
    assert_solve_conformance("quip-cpu-flatiron").await;
}

/// The identities with a beta ladder list `linear-schedule`; the Simulated
/// Bifurcation ones have none and do not. Every identity states its limits.
#[test]
fn capabilities_advertise_the_schedule_feature_and_limits() {
    let bins = ["quip-cpu-sa", "quip-cpu-gibbs", "quip-cpu-sb"];
    ensure_built(&bins);
    for bin in bins {
        let out = Command::new(profile_bin(bin))
            .arg("--capabilities")
            .output()
            .unwrap();
        let caps: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        let linear = caps["features"]
            .as_array()
            .is_some_and(|f| f.iter().any(|f| f == "linear-schedule"));
        assert_eq!(linear, bin != "quip-cpu-sb", "{bin}: {caps}");
        assert_eq!(caps["maxDegree"], 0, "{bin}: {caps}");
        assert_eq!(caps["maxLocalFieldMilli"], 0, "{bin}: {caps}");
        let resolution = if bin == "quip-cpu-sb" {
            serde_json::Value::Null
        } else {
            serde_json::json!(1)
        };
        assert_eq!(caps["exactResolutionMilli"], resolution, "{bin}: {caps}");
    }
}

/// Cross-package binary path (deps/ → profile/ → bin).
fn profile_bin(name: &str) -> String {
    let name = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    let mut p = std::env::current_exe().expect("test exe path");
    p.pop(); // deps/
    p.pop(); // <profile>/
    p.push(&name);
    p.to_string_lossy().into_owned()
}

/// Build the crate and assert the named binaries landed in the profile
/// directory. `features` is forwarded to `cargo build --features`; the
/// experimental SB and tensor-network binaries need it, the production ones do
/// not.
fn ensure_built_with(package_bins: &[&str], features: &[&str]) {
    let mut args: Vec<String> = vec!["build".into(), "-p".into(), "quip-solver-cpu".into()];
    if !features.is_empty() {
        args.push("--features".into());
        args.push(features.join(","));
    }
    let status = Command::new(env!("CARGO"))
        .args(&args)
        .status()
        .expect("cargo build quip-solver-cpu");
    assert!(status.success(), "failed to build quip-solver-cpu");
    for b in package_bins {
        assert!(
            std::path::Path::new(&profile_bin(b)).exists(),
            "missing binary {b} at {}",
            profile_bin(b)
        );
    }
}

fn ensure_built(package_bins: &[&str]) {
    ensure_built_with(package_bins, &[]);
}

#[tokio::test]
async fn quip_cpu_sa_passes_conformance() {
    ensure_built(&["quip-cpu-sa"]);
    let miner = profile_bin("quip-cpu-sa");
    let socket = format!(
        "/tmp/quip-cpu-sa-conf-{}-{}.sock",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let report = drive_miner_bounded(&miner, &format!("unix://{socket}")).await;
    assert!(report.handshake_ok, "SA handshake failed");
    assert_eq!(
        report.result_job_ids().len(),
        5,
        "expected 5 job results (job-1, job-2, job-hash, job-sparse, job-seeded)"
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-seeded"),
        "missing result for seeded job-seeded: {:?}",
        report.result_job_ids()
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-1"),
        "missing result for job-1: {:?}",
        report.result_job_ids()
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-2"),
        "missing result for job-2: {:?}",
        report.result_job_ids()
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-hash"),
        "missing result for topology-hash job-hash: {:?}",
        report.result_job_ids()
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-sparse"),
        "missing result for sparse-topology job-sparse: {:?}",
        report.result_job_ids()
    );
    assert!(
        report.has_reject(b"job-bad-h", RejectReason::Malformed),
        "missing MALFORMED reject for job-bad-h: {:?}",
        report.rejects
    );
    assert!(
        report.has_reject(b"job-gate", RejectReason::UnsupportedKind),
        "missing UNSUPPORTED_KIND reject for job-gate: {:?}",
        report.rejects
    );
    assert!(
        report.has_reject(b"job-old", RejectReason::Expired),
        "missing EXPIRED reject for job-old: {:?}",
        report.rejects
    );
    assert_eq!(report.exit_code, 0, "clean shutdown expected");
}

#[tokio::test]
async fn quip_cpu_gibbs_passes_conformance() {
    ensure_built(&["quip-cpu-gibbs"]);
    let miner = profile_bin("quip-cpu-gibbs");
    let socket = format!(
        "/tmp/quip-cpu-gibbs-conf-{}-{}.sock",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let report = drive_miner_bounded(&miner, &format!("unix://{socket}")).await;
    assert!(report.handshake_ok, "Gibbs handshake failed");
    assert_eq!(
        report.result_job_ids().len(),
        5,
        "expected 5 job results (job-1, job-2, job-hash, job-sparse, job-seeded)"
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-seeded"),
        "missing result for seeded job-seeded: {:?}",
        report.result_job_ids()
    );
    assert!(report.result_job_ids().iter().any(|id| id == b"job-1"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-2"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-hash"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-sparse"));
    assert!(report.has_reject(b"job-bad-h", RejectReason::Malformed));
    assert!(report.has_reject(b"job-gate", RejectReason::UnsupportedKind));
    assert!(report.has_reject(b"job-old", RejectReason::Expired));
    assert_eq!(report.exit_code, 0, "clean shutdown expected");
}

#[tokio::test]
async fn quip_cpu_sb_passes_conformance() {
    ensure_built(&["quip-cpu-sb"]);
    let miner = profile_bin("quip-cpu-sb");
    let socket = format!(
        "/tmp/quip-cpu-sb-conf-{}-{}.sock",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let report = drive_miner_bounded(&miner, &format!("unix://{socket}")).await;
    assert!(report.handshake_ok, "SB handshake failed");
    assert_eq!(
        report.result_job_ids().len(),
        5,
        "expected 5 job results (job-1, job-2, job-hash, job-sparse, job-seeded)"
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-seeded"),
        "missing result for seeded job-seeded: {:?}",
        report.result_job_ids()
    );
    assert!(report.result_job_ids().iter().any(|id| id == b"job-1"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-2"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-hash"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-sparse"));
    assert!(report.has_reject(b"job-bad-h", RejectReason::Malformed));
    assert!(report.has_reject(b"job-gate", RejectReason::UnsupportedKind));
    assert!(report.has_reject(b"job-old", RejectReason::Expired));
    assert_eq!(report.exit_code, 0, "clean shutdown expected");
}

#[test]
fn capabilities_and_version_and_check() {
    ensure_built(&["quip-cpu-sa", "quip-cpu-gibbs", "quip-cpu-sb"]);

    for (bin, algo) in [
        ("quip-cpu-sa", "sa"),
        ("quip-cpu-gibbs", "gibbs"),
        ("quip-cpu-sb", "sb"),
    ] {
        let path = profile_bin(bin);

        let out = Command::new(&path).arg("--capabilities").output().unwrap();
        assert!(out.status.success(), "{bin} --capabilities failed");
        let s = String::from_utf8(out.stdout).unwrap();
        assert!(s.contains("\"backend\":\"cpu\""), "{bin}: {s}");
        assert!(
            s.contains(&format!("\"algorithm\":\"{algo}\"")),
            "{bin}: {s}"
        );

        let out = Command::new(&path).arg("--version").output().unwrap();
        assert!(out.status.success());
        assert!(String::from_utf8(out.stdout).unwrap().contains("protocol"));

        assert!(Command::new(&path)
            .arg("--check")
            .status()
            .unwrap()
            .success());
    }
}

/// Ballistic SB drives the same protocol surface as the production binaries.
/// Gated so a default `cargo test` neither builds nor exercises the
/// experimental track.
#[cfg(feature = "experimental")]
#[tokio::test]
async fn quip_cpu_bsb_passes_conformance() {
    ensure_built_with(&["quip-cpu-bsb"], &["experimental"]);
    let miner = profile_bin("quip-cpu-bsb");
    let socket = format!(
        "/tmp/quip-cpu-bsb-conf-{}-{}.sock",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let report = drive_miner_bounded(&miner, &format!("unix://{socket}")).await;
    assert!(report.handshake_ok, "bsb handshake failed");
    assert_eq!(
        report.result_job_ids().len(),
        5,
        "expected 5 job results (job-1, job-2, job-hash, job-sparse, job-seeded)"
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-seeded"),
        "missing result for seeded job-seeded: {:?}",
        report.result_job_ids()
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-1"),
        "missing result for job-1: {:?}",
        report.result_job_ids()
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-2"),
        "missing result for job-2: {:?}",
        report.result_job_ids()
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-hash"),
        "missing result for topology-hash job-hash: {:?}",
        report.result_job_ids()
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-sparse"),
        "missing result for sparse-topology job-sparse: {:?}",
        report.result_job_ids()
    );
    assert!(
        report.has_reject(b"job-bad-h", RejectReason::Malformed),
        "missing MALFORMED reject for job-bad-h: {:?}",
        report.rejects
    );
    assert!(
        report.has_reject(b"job-gate", RejectReason::UnsupportedKind),
        "missing UNSUPPORTED_KIND reject for job-gate: {:?}",
        report.rejects
    );
    assert!(
        report.has_reject(b"job-old", RejectReason::Expired),
        "missing EXPIRED reject for job-old: {:?}",
        report.rejects
    );
    assert_eq!(report.exit_code, 0, "clean shutdown expected");
}

/// Every experimental binary and the algorithm string it must report. The
/// capabilities test below drives every row, so adding a binary is one line. This stays scoped to the
/// SB-kernel variants; mfa and mps get their own coverage.
///
/// A slice rather than an array so the loop below reads the same at one row as
/// at four.
#[cfg(feature = "experimental")]
const EXPERIMENTAL_BINS: &[(&str, &str)] = &[
    ("quip-cpu-bsb", "bsb"),
    ("quip-cpu-hdsb", "hdsb"),
    ("quip-cpu-hbsb", "hbsb"),
    ("quip-cpu-gbsb", "gbsb"),
    ("quip-cpu-gdsb", "gdsb"),
    ("quip-cpu-tedsb", "tedsb"),
    ("quip-cpu-sbqa", "sbqa"),
    ("quip-cpu-ggdsb", "ggdsb"),
    ("quip-cpu-fsa", "fsa"),
    ("quip-cpu-msa", "msa"),
    ("quip-cpu-mps", "mps"),
    ("quip-cpu-mfa", "mfa"),
    ("quip-cpu-flatiron", "flatiron"),
];

/// The capabilities, version, and self-check surface of the experimental
/// binaries.
#[cfg(feature = "experimental")]
#[test]
fn experimental_capabilities_and_version_and_check() {
    let bins: Vec<&str> = EXPERIMENTAL_BINS.iter().map(|&(bin, _)| bin).collect();
    ensure_built_with(&bins, &["experimental"]);

    for &(bin, algo) in EXPERIMENTAL_BINS {
        let path = profile_bin(bin);

        let out = Command::new(&path).arg("--capabilities").output().unwrap();
        assert!(out.status.success(), "{bin} --capabilities failed");
        let s = String::from_utf8(out.stdout).unwrap();
        assert!(s.contains("\"backend\":\"cpu\""), "{bin}: {s}");
        assert!(
            s.contains(&format!("\"algorithm\":\"{algo}\"")),
            "{bin}: {s}"
        );

        let out = Command::new(&path).arg("--version").output().unwrap();
        assert!(out.status.success());
        assert!(String::from_utf8(out.stdout).unwrap().contains("protocol"));

        assert!(Command::new(&path)
            .arg("--check")
            .status()
            .unwrap()
            .success());
    }
}

/// Heated discrete SB drives the same protocol surface as the production
/// binaries. The heating term draws no random numbers, so this is as
/// deterministic as the unheated variants.
#[cfg(feature = "experimental")]
#[tokio::test]
async fn quip_cpu_hdsb_passes_conformance() {
    ensure_built_with(&["quip-cpu-hdsb"], &["experimental"]);
    let miner = profile_bin("quip-cpu-hdsb");
    let socket = format!(
        "/tmp/quip-cpu-hdsb-conf-{}-{}.sock",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let report = drive_miner_bounded(&miner, &format!("unix://{socket}")).await;
    assert!(report.handshake_ok, "HDSB handshake failed");
    assert_eq!(
        report.result_job_ids().len(),
        5,
        "expected 5 job results (job-1, job-2, job-hash, job-sparse, job-seeded)"
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-seeded"),
        "missing result for seeded job-seeded: {:?}",
        report.result_job_ids()
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-1"),
        "missing result for job-1: {:?}",
        report.result_job_ids()
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-2"),
        "missing result for job-2: {:?}",
        report.result_job_ids()
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-hash"),
        "missing result for topology-hash job-hash: {:?}",
        report.result_job_ids()
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-sparse"),
        "missing result for sparse-topology job-sparse: {:?}",
        report.result_job_ids()
    );
    assert!(
        report.has_reject(b"job-bad-h", RejectReason::Malformed),
        "missing MALFORMED reject for job-bad-h: {:?}",
        report.rejects
    );
    assert!(
        report.has_reject(b"job-gate", RejectReason::UnsupportedKind),
        "missing UNSUPPORTED_KIND reject for job-gate: {:?}",
        report.rejects
    );
    assert!(
        report.has_reject(b"job-old", RejectReason::Expired),
        "missing EXPIRED reject for job-old: {:?}",
        report.rejects
    );
    assert_eq!(report.exit_code, 0, "clean shutdown expected");
}

/// Heated ballistic SB drives the same protocol surface as the production
/// binaries. This is the strongest heating of the four SB variants, so it is
/// the variant most able to run away numerically; the wall invariant is
/// guarded in the kernel's own tests.
#[cfg(feature = "experimental")]
#[tokio::test]
async fn quip_cpu_hbsb_passes_conformance() {
    ensure_built_with(&["quip-cpu-hbsb"], &["experimental"]);
    let miner = profile_bin("quip-cpu-hbsb");
    let socket = format!(
        "/tmp/quip-cpu-hbsb-conf-{}-{}.sock",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let report = drive_miner_bounded(&miner, &format!("unix://{socket}")).await;
    assert!(report.handshake_ok, "HBSB handshake failed");
    assert_eq!(
        report.result_job_ids().len(),
        5,
        "expected 5 job results (job-1, job-2, job-hash, job-sparse, job-seeded)"
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-seeded"),
        "missing result for seeded job-seeded: {:?}",
        report.result_job_ids()
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-1"),
        "missing result for job-1: {:?}",
        report.result_job_ids()
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-2"),
        "missing result for job-2: {:?}",
        report.result_job_ids()
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-hash"),
        "missing result for topology-hash job-hash: {:?}",
        report.result_job_ids()
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-sparse"),
        "missing result for sparse-topology job-sparse: {:?}",
        report.result_job_ids()
    );
    assert!(
        report.has_reject(b"job-bad-h", RejectReason::Malformed),
        "missing MALFORMED reject for job-bad-h: {:?}",
        report.rejects
    );
    assert!(
        report.has_reject(b"job-gate", RejectReason::UnsupportedKind),
        "missing UNSUPPORTED_KIND reject for job-gate: {:?}",
        report.rejects
    );
    assert!(
        report.has_reject(b"job-old", RejectReason::Expired),
        "missing EXPIRED reject for job-old: {:?}",
        report.rejects
    );
    assert_eq!(report.exit_code, 0, "clean shutdown expected");
}

/// The tensor-network miner drives the same protocol surface. On the mock
/// coordinator's small graphs the bond dimension is well above 1, so this
/// exercises the real zip-up rather than only the product-state path.
#[cfg(feature = "experimental")]
#[tokio::test]
async fn quip_cpu_mps_passes_conformance() {
    ensure_built_with(&["quip-cpu-mps"], &["experimental"]);
    let miner = profile_bin("quip-cpu-mps");
    let socket = format!(
        "/tmp/quip-cpu-mps-conf-{}-{}.sock",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let report = drive_miner_bounded(&miner, &format!("unix://{socket}")).await;
    assert!(report.handshake_ok, "mps handshake failed");
    assert_eq!(
        report.result_job_ids().len(),
        5,
        "expected 5 job results (job-1, job-2, job-hash, job-sparse, job-seeded)"
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-seeded"),
        "missing result for seeded job-seeded: {:?}",
        report.result_job_ids()
    );
    assert!(report.result_job_ids().iter().any(|id| id == b"job-1"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-2"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-hash"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-sparse"));
    assert!(report.has_reject(b"job-bad-h", RejectReason::Malformed));
    assert!(report.has_reject(b"job-gate", RejectReason::UnsupportedKind));
    assert!(report.has_reject(b"job-old", RejectReason::Expired));
    assert_eq!(report.exit_code, 0, "clean shutdown expected");
}

/// The BP-TNS miner drives the same protocol surface. On the mock
/// coordinator's small graphs `select_chi` resolves above 1, so this
/// exercises the gauged gate, the BP pass, and conditioned sampling rather
/// than only the mean-field path.
#[cfg(feature = "experimental")]
#[tokio::test]
async fn quip_cpu_flatiron_passes_conformance() {
    ensure_built_with(&["quip-cpu-flatiron"], &["experimental"]);
    let miner = profile_bin("quip-cpu-flatiron");
    let socket = format!(
        "/tmp/quip-cpu-flatiron-conf-{}-{}.sock",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let report = drive_miner_bounded(&miner, &format!("unix://{socket}")).await;
    assert!(report.handshake_ok, "flatiron handshake failed");
    assert_eq!(
        report.result_job_ids().len(),
        5,
        "expected 5 job results (job-1, job-2, job-hash, job-sparse, job-seeded)"
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-seeded"),
        "missing result for seeded job-seeded: {:?}",
        report.result_job_ids()
    );
    assert!(report.result_job_ids().iter().any(|id| id == b"job-1"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-2"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-hash"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-sparse"));
    assert!(report.has_reject(b"job-bad-h", RejectReason::Malformed));
    assert!(report.has_reject(b"job-gate", RejectReason::UnsupportedKind));
    assert!(report.has_reject(b"job-old", RejectReason::Expired));
    assert_eq!(report.exit_code, 0, "clean shutdown expected");
}

/// Mean-field annealing drives the same protocol surface. It is the same
/// kernel as `quip-cpu-mps` pinned to bond dimension 1, so this covers the
/// product-state path end to end through the coordinator.
#[cfg(feature = "experimental")]
#[tokio::test]
async fn quip_cpu_mfa_passes_conformance() {
    ensure_built_with(&["quip-cpu-mfa"], &["experimental"]);
    let miner = profile_bin("quip-cpu-mfa");
    let socket = format!(
        "/tmp/quip-cpu-mfa-conf-{}-{}.sock",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let report = drive_miner_bounded(&miner, &format!("unix://{socket}")).await;
    assert!(report.handshake_ok, "mfa handshake failed");
    assert_eq!(
        report.result_job_ids().len(),
        5,
        "expected 5 job results (job-1, job-2, job-hash, job-sparse, job-seeded)"
    );
    assert!(
        report.result_job_ids().iter().any(|id| id == b"job-seeded"),
        "missing result for seeded job-seeded: {:?}",
        report.result_job_ids()
    );
    assert!(report.result_job_ids().iter().any(|id| id == b"job-1"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-2"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-hash"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-sparse"));
    assert!(report.has_reject(b"job-bad-h", RejectReason::Malformed));
    assert!(report.has_reject(b"job-gate", RejectReason::UnsupportedKind));
    assert!(report.has_reject(b"job-old", RejectReason::Expired));
    assert_eq!(report.exit_code, 0, "clean shutdown expected");
}
