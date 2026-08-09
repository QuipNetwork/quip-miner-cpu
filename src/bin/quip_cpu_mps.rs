//! CPU tensor-network miner (`quip-cpu-mps`), experimental.
//!
//! Build with `cargo build --release --features experimental`. The bond
//! dimension is chosen per job from a deterministic budget and capped at 32;
//! on the production topology that cap resolves to 1, which is mean-field
//! annealing. `QUIP_MPS_INIT=random` replaces the anneal with uniform random
//! starting configurations, which is the H3 experiment's control arm.

use clap::Parser;
use quip_miner_core::{run, CommonArgs};
use quip_miner_cpu::{MpsConfig, MpsSampler, CPU_MPS_IDENTITY};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run(CPU_MPS_IDENTITY, &cli.common, || {
        Ok(MpsSampler::new(MpsConfig::from_env(32)))
    })
}
