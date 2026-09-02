//! CPU replica-ring simulated-bifurcation miner (`quip-cpu-sbqa`).

use clap::Parser;
use quip_miner_cpu::{SbSampler, SbqaConfig, CPU_SBQA_IDENTITY};
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
    run(CPU_SBQA_IDENTITY, &cli.common, || {
        Ok(SbSampler::ring(SbqaConfig::default()))
    })
}
