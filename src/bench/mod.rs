//! Benchmark harness for the CPU Ising sampler.
//!
//! Runs a fixed model (or a corpus) through [`crate::sample_ising`] under a
//! `tracing` subscriber, aggregates per-part busy-time, derives per-spin and
//! per-flip costs (aggregate ÷ frequency), and emits per-part JSON plus
//! `tracing-flame` folded stacks for a flame graph.

pub mod report;
pub mod source;
pub mod timing;

/// Parsed bench invocation (populated by the CLI in Task 6/7).
#[derive(Debug, Clone)]
pub struct BenchArgs {
    /// Warm-up iterations discarded before measurement.
    pub warmup: usize,
    /// Measured iterations per model.
    pub iters: usize,
}

/// Bench failure with a human-actionable message.
#[derive(Debug)]
pub enum BenchError {
    /// I/O failure writing a report or folded-stack file.
    Io(String),
    /// Model source could not be built (message names the cause).
    Source(String),
    /// JSON serialization failure.
    Serialize(String),
}

impl std::fmt::Display for BenchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(m) => write!(f, "bench i/o error: {m}"),
            Self::Source(m) => write!(f, "bench model source error: {m}"),
            Self::Serialize(m) => write!(f, "bench serialize error: {m}"),
        }
    }
}

impl std::error::Error for BenchError {}

/// Entry point (fleshed out in Task 7). Currently a no-op success.
pub fn run_bench(_args: &BenchArgs) -> Result<(), BenchError> {
    Ok(())
}
