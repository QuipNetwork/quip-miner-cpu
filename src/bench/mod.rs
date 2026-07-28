//! Benchmark harness for the CPU Ising sampler.
//!
//! Runs a fixed model (or a corpus) through [`crate::sample_ising`] under a
//! `tracing` subscriber, aggregates per-part busy-time, derives per-spin and
//! per-flip costs (aggregate ÷ frequency), and emits per-part JSON plus
//! `tracing-flame` folded stacks for a flame graph.

pub mod report;
pub mod source;
pub mod timing;

use std::path::Path;
use std::sync::Arc;

use tracing_subscriber::prelude::*;

use crate::bench::timing::{with_active_agg, TimingAggregator, TimingLayer};

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

/// Run `f` under a subscriber that both aggregates per-part busy-time and writes
/// `tracing-flame` folded stacks to `folded_path`. The subscriber is installed
/// only for the duration of `f` (`with_default`), so it never leaks into other
/// tests or a session process.
pub fn run_instrumented<R>(
    agg: &Arc<TimingAggregator>,
    folded_path: &Path,
    f: impl FnOnce() -> R,
) -> Result<R, BenchError> {
    let (flame, guard) = tracing_flame::FlameLayer::with_file(folded_path)
        .map_err(|e| BenchError::Io(format!("open folded file {}: {e}", folded_path.display())))?;
    let timing = TimingLayer::new(Arc::clone(agg));
    let subscriber = tracing_subscriber::registry().with(timing).with(flame);
    let out = tracing::subscriber::with_default(subscriber, || with_active_agg(agg, f));
    // Flush folded stacks to disk before returning.
    guard
        .flush()
        .map_err(|e| BenchError::Io(format!("flush folded stacks: {e}")))?;
    drop(guard);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::timing::TimingAggregator;
    use std::sync::Arc;

    #[test]
    fn instrumented_run_writes_folded_and_aggregates() {
        let dir = std::env::temp_dir().join(format!("quipbench-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let folded = dir.join("stacks.folded");
        let agg = Arc::new(TimingAggregator::default());
        let g = crate::IsingGraph::new(vec![0.0, 0.0], vec![-1.0], vec![(0, 1)]);
        let params = crate::SampleParams {
            num_reads: 2,
            num_sweeps: 16,
            seed: 1,
            ..Default::default()
        };
        run_instrumented(&agg, &folded, || {
            let _ = crate::sample_ising(&g, &params, crate::Algorithm::Sa);
        })
        .expect("instrumented run");
        let snap = agg.snapshot();
        assert!(snap.contains_key("anneal_read"), "seam span recorded");
        assert!(snap.contains_key("sweep_loop"), "child span recorded");
        let folded_txt = std::fs::read_to_string(&folded).expect("folded file");
        assert!(!folded_txt.trim().is_empty(), "folded stacks non-empty");
        assert!(folded_txt.contains("anneal_read"), "folded names present");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
