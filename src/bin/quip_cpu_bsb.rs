//! CPU ballistic simulated-bifurcation miner (`quip-cpu-bsb`).

use clap::Parser;
use quip_miner_core::{run, CommonArgs};
use quip_miner_cpu::{SbSampler, BSB, CPU_BSB_IDENTITY};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run(CPU_BSB_IDENTITY, &cli.common, || Ok(SbSampler::new(BSB)))
}
