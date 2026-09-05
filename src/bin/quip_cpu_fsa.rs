//! CPU tabulated-threshold annealing miner (`quip-cpu-fsa`).

use clap::Parser;
use quip_miner_cpu::{SaSampler, SaVariant, CPU_FSA_IDENTITY};
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
    run(CPU_FSA_IDENTITY, &cli.common, || {
        Ok(SaSampler::new(SaVariant::Tabulated))
    })
}
