//! CPU generalized discrete simulated-bifurcation miner (`quip-cpu-gdsb`):
//! dSB with edge-of-chaos control, this project's extension of the published
//! ballistic form.

use clap::Parser;
use quip_miner_cpu::{SbSampler, CPU_GDSB_IDENTITY, GDSB};
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
    run(CPU_GDSB_IDENTITY, &cli.common, || Ok(SbSampler::new(GDSB)))
}
