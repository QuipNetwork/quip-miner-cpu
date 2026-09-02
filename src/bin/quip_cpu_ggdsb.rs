//! CPU globally guided discrete simulated-bifurcation miner (`quip-cpu-ggdsb`).

use clap::Parser;
use quip_miner_cpu::{GgsbConfig, SbSampler, CPU_GGDSB_IDENTITY, DSB};
use quip_solver_core::{run, CommonArgs};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run(CPU_GGDSB_IDENTITY, &cli.common, || {
        Ok(SbSampler::swarm(DSB, GgsbConfig::default()))
    })
}
