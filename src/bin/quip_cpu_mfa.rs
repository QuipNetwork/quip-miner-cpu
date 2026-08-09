//! CPU mean-field annealing miner (`quip-cpu-mfa`): the tensor-network
//! kernel with bond dimension fixed at 1.
//!
//! At bond dimension 1 the state is a product state, so the coupling gate is a
//! rank-1 truncation of one 2x2 matrix per edge and the chain span drops out of
//! the cost. This is the regime `quip-cpu-mps` already falls into on any graph
//! as wide as the production topology; naming it as its own binary lets a
//! campaign measure it directly.
//!
//! `QUIP_MPS_INIT=random` replaces the anneal with uniform random starting
//! configurations. Both arms share the sampler and the greedy polish, which is
//! what makes the H3 seeding comparison meaningful.

use clap::Parser;
use quip_miner_core::{run, CommonArgs};
use quip_miner_cpu::{MpsConfig, MpsSampler, CPU_MFA_IDENTITY};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run(CPU_MFA_IDENTITY, &cli.common, || {
        Ok(MpsSampler::new(MpsConfig::from_env(1)))
    })
}
