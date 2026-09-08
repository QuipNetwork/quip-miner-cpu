//! CPU generalized ballistic simulated-bifurcation miner (`quip-cpu-gbsb`):
//! bSB with edge-of-chaos control of each particle's bifurcation parameter.

use clap::Parser;
use quip_miner_cpu::{SbSampler, CPU_GBSB_IDENTITY, GBSB};
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
    run(CPU_GBSB_IDENTITY, &cli.common, || Ok(SbSampler::new(GBSB)))
}
