//! CPU heated ballistic simulated-bifurcation miner (`quip-cpu-hbsb`).

use clap::Parser;
use quip_miner_core::{run, CommonArgs};
use quip_miner_cpu::{SbSampler, CPU_HBSB_IDENTITY, HBSB};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run(CPU_HBSB_IDENTITY, &cli.common, || Ok(SbSampler::new(HBSB)))
}
