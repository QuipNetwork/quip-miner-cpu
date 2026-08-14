//! Protocol conformance: spawn SA and Gibbs miners against quip-mock-coordinator.

use quip_mock_coordinator::driver::{drive_miner, DriverReport};
use quip_proto::v1::RejectReason;
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
    let mut args: Vec<String> = vec!["build".into(), "-p".into(), "quip-miner-cpu".into()];
    if !features.is_empty() {
        args.push("--features".into());
        args.push(features.join(","));
    }
    let status = Command::new(env!("CARGO"))
        .args(&args)
        .status()
        .expect("cargo build quip-miner-cpu");
    assert!(status.success(), "failed to build quip-miner-cpu");
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
        3,
        "expected 3 job results (job-1, job-2, job-hash)"
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
        3,
        "expected 3 job results (job-1, job-2, job-hash)"
    );
    assert!(report.result_job_ids().iter().any(|id| id == b"job-1"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-2"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-hash"));
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
        3,
        "expected 3 job results (job-1, job-2, job-hash)"
    );
    assert!(report.result_job_ids().iter().any(|id| id == b"job-1"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-2"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-hash"));
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
        3,
        "expected 3 job results (job-1, job-2, job-hash)"
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
        3,
        "expected 3 job results (job-1, job-2, job-hash)"
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
        3,
        "expected 3 job results (job-1, job-2, job-hash)"
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
        3,
        "expected 3 job results (job-1, job-2, job-hash)"
    );
    assert!(report.result_job_ids().iter().any(|id| id == b"job-1"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-2"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-hash"));
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
        3,
        "expected 3 job results (job-1, job-2, job-hash)"
    );
    assert!(report.result_job_ids().iter().any(|id| id == b"job-1"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-2"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-hash"));
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
        3,
        "expected 3 job results (job-1, job-2, job-hash)"
    );
    assert!(report.result_job_ids().iter().any(|id| id == b"job-1"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-2"));
    assert!(report.result_job_ids().iter().any(|id| id == b"job-hash"));
    assert!(report.has_reject(b"job-bad-h", RejectReason::Malformed));
    assert!(report.has_reject(b"job-gate", RejectReason::UnsupportedKind));
    assert!(report.has_reject(b"job-old", RejectReason::Expired));
    assert_eq!(report.exit_code, 0, "clean shutdown expected");
}
