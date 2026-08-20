//! Process-spawn exit-code parity with quip-mock-miner (see
//! rust/quip-mock-miner/tests/handshake.rs): quip-solver-core-backed binaries
//! must exit the same documented codes (64/77) as the reference mock.

use std::process::Command;

/// Every production binary in this crate. The 64 and 77 exit codes come from
/// `quip-solver-core`, so each binary inherits them, but each one must still be
/// named here or a new binary could ship without the parity check.
fn bins() -> [&'static str; 2] {
    [
        env!("CARGO_BIN_EXE_quip-cpu-sa"),
        env!("CARGO_BIN_EXE_quip-cpu-sb"),
    ]
}

#[test]
fn missing_coordinator_exits_64_not_panic() {
    for bin in bins() {
        let out = Command::new(bin)
            .env("QUIP_SESSION_TOKEN", "tok")
            .output()
            .unwrap();
        // No --quip-coordinator and no --capabilities/--check → ConfigInvalid (64).
        assert_eq!(
            out.status.code(),
            Some(64),
            "{bin}: missing --quip-coordinator must exit 64 (got {:?}, stderr={})",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn missing_session_token_exits_77() {
    for bin in bins() {
        let out = Command::new(bin)
            .arg("--quip-coordinator")
            .arg("unix:///tmp/quip-no-such-socket.sock")
            .env_remove("QUIP_SESSION_TOKEN")
            .output()
            .unwrap();
        assert_eq!(
            out.status.code(),
            Some(77),
            "{bin}: missing QUIP_SESSION_TOKEN must exit 77 (got {:?}, stderr={})",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

/// `--log-level` is validated by `quip-solver-core`'s `CommonArgs` clap parser
/// against a fixed list (`trace`, `debug`, `info`, `warn`, `error`), so an
/// unknown level is rejected before `--capabilities` is handled and before
/// `logging::init` ever runs. Clap's usage-error exit code is 2, not the
/// sysexits `ConfigInvalid` (64) a runtime rejection would use.
///
/// A core revision that predates parse-time validation never rejects the
/// level and exits 0 here, which is exactly the regression this test guards.
#[test]
fn invalid_log_level_rejected_at_parse_time() {
    for bin in bins() {
        let out = Command::new(bin)
            .arg("--capabilities")
            .arg("--log-level")
            .arg("bogus")
            .env("QUIP_SESSION_TOKEN", "tok")
            .output()
            .unwrap();
        assert_eq!(
            out.status.code(),
            Some(2),
            "{bin}: an unknown --log-level must be rejected at parse time (got {:?}, stdout={}, stderr={})",
            out.status.code(),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("invalid value 'bogus'") && stderr.contains("--log-level"),
            "{bin}: stderr must name the bad level, got {stderr}"
        );
    }
}
