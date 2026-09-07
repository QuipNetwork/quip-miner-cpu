//! CPU multi-spin coded annealing miner (`quip-cpu-msa`).

use clap::Parser;
use quip_miner_cpu::{SaSampler, SaVariant, CPU_MSA_IDENTITY};
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
    run(CPU_MSA_IDENTITY, &cli.common, || {
        Ok(SaSampler::new(SaVariant::MultiSpin))
    })
}
