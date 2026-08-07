//! CPU chromatic Gibbs miner (`quip-cpu-gibbs`).

use clap::Parser;
use quip_miner_core::{run, CommonArgs};
use quip_miner_cpu::{Algorithm, CpuSampler, GibbsConfig, CPU_GIBBS_IDENTITY};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    /// Workers splitting each colour class. The default is the measured best
    /// setting; higher counts scale worse and oversubscription is refused.
    #[arg(long, default_value_t = quip_miner_cpu::gibbs_parallel::DEFAULT_GIBBS_WORKERS)]
    gibbs_workers: usize,

    /// Refuse any graph needing more than this many colour classes.
    #[arg(long)]
    gibbs_max_colors: Option<usize>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let gibbs = GibbsConfig {
        workers: cli.gibbs_workers,
        max_colors: cli.gibbs_max_colors,
    };
    // Checked before the session opens. A miner that runs at a worker count the
    // host cannot serve does not degrade gently: the class barrier spins, so an
    // oversubscribed run collapses. Exit as a configuration error instead.
    if let Err(e) = gibbs.validate() {
        #[expect(
            clippy::print_stderr,
            reason = "a startup configuration error must reach the operator, and the \
                      session loop that would otherwise report it never opens"
        )]
        {
            eprintln!("configuration error: {e}");
        }
        return ExitCode::from(64);
    }
    run(CPU_GIBBS_IDENTITY, &cli.common, || {
        Ok(CpuSampler::new(Algorithm::Gibbs).with_gibbs_config(gibbs))
    })
}
