//! CPU BP-TNS miner (`quip-cpu-bptns`), experimental.
//!
//! Build with `cargo build --release --features experimental`. The bond
//! dimension is chosen per job from the memory and flop budgets and capped
//! at 8; on the production topology the vertex degrees resolve that cap to
//! 1, which is mean-field annealing with BP bookkeeping. This binary exists
//! to measure exactly that degradation against the real corpora.

use clap::Parser;
use quip_miner_core::{run, CommonArgs};
use quip_miner_cpu::{BptnsConfig, BptnsSampler, CPU_BPTNS_IDENTITY};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run(CPU_BPTNS_IDENTITY, &cli.common, || {
        Ok(BptnsSampler::new(BptnsConfig::new(8)))
    })
}
