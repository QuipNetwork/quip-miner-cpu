//! Process-spawn exit-code parity with quip-mock-miner (see
//! rust/quip-mock-miner/tests/handshake.rs): quip-miner-core-backed binaries
//! must exit the same documented codes (64/77) as the reference mock.

use std::process::Command;

/// Every production binary in this crate. The 64 and 77 exit codes come from
/// `quip-miner-core`, so each binary inherits them, but each one must still be
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
