//! CPU heated discrete simulated-bifurcation miner (`quip-cpu-hdsb`).

use clap::Parser;
use quip_miner_core::{run, CommonArgs};
use quip_miner_cpu::{SbSampler, CPU_HDSB_IDENTITY, HDSB};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run(CPU_HDSB_IDENTITY, &cli.common, || Ok(SbSampler::new(HDSB)))
}
