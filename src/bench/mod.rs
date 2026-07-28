//! Benchmark harness for the CPU Ising sampler.
//!
//! Runs a fixed model (or a corpus) through [`crate::sample_ising`] under a
//! `tracing` subscriber, aggregates per-part busy-time, derives per-spin and
//! per-flip costs (aggregate ÷ frequency), and emits per-part JSON plus
//! `tracing-flame` folded stacks for a flame graph.
//!
//! The headline per-part JSON always comes from the default build: coarse
//! seam spans only (`cpu_graph_build`, `beta_schedule`, `anneal_read`,
//! `random_spins`, `seed_heff`, `sweep_loop`, `score`), entered O(reads) times
//! per model. Building with `--features fine-spans` additionally spans every
//! spin decision and accepted flip inside the hot loop (`spin_decision`,
//! `apply_flip`) for a diagnostic cross-check pass — never for the headline
//! numbers, since millions of span enters per read measurably inflate the
//! total wall time they are meant to describe. Use `fine-spans` output only
//! for relative attribution and to sanity-check `derived.per_spin_ns` against
//! an external `perf`/`cargo-flamegraph` run, e.g.:
//! `cargo flamegraph --bin quip-cpu-bench -- --nodes 512 --edges 2048`.

pub mod report;
pub mod source;
pub mod timing;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tracing_subscriber::prelude::*;

use crate::bench::report::{build_report, ModelDescriptor, ReportInputs};
use crate::bench::source::{from_jsonl, synthetic, ModelSpec, Topology};
use crate::bench::timing::{with_active_agg, TimingAggregator, TimingLayer};
use crate::{sample_ising, Algorithm, SampleParams};

/// Where the bench pulls models from.
#[derive(Debug, Clone)]
pub enum SourceKind {
    /// Generate one random model of this shape.
    Synthetic {
        /// Variable count.
        n_nodes: usize,
        /// Edge count (best effort; duplicate/degenerate draws are skipped).
        n_edges: usize,
    },
    /// Read a corpus JSONL; `manifest` supplies topology for nonce-refs.
    Corpus {
        /// Path to the JSONL corpus.
        path: PathBuf,
        /// Manifest supplying topology for nonce-ref entries.
        manifest: Option<PathBuf>,
    },
}

/// Fully parsed bench invocation.
#[derive(Debug, Clone)]
pub struct BenchArgs {
    /// Sampler algorithm.
    pub algorithm: Algorithm,
    /// Model source.
    pub source: SourceKind,
    /// Reads per sample call.
    pub num_reads: usize,
    /// Sweeps per read.
    pub num_sweeps: usize,
    /// Sweeps per beta rung.
    pub sweeps_per_beta: usize,
    /// PRNG seed.
    pub seed: u64,
    /// Warm-up iterations (discarded).
    pub warmup: usize,
    /// Measured iterations.
    pub iters: usize,
    /// Output directory for `<model_id>.json` and `.folded`.
    pub out_dir: PathBuf,
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

fn algorithm_name(a: Algorithm) -> &'static str {
    match a {
        Algorithm::Sa => "sa",
        Algorithm::Gibbs => "gibbs",
    }
}

fn load_topology(manifest: Option<&PathBuf>) -> Result<Option<Topology>, BenchError> {
    let Some(path) = manifest else {
        return Ok(None);
    };
    let text = std::fs::read_to_string(path)
        .map_err(|e| BenchError::Source(format!("read manifest {}: {e}", path.display())))?;
    let topo = source::topology_from_manifest_json(&text)
        .map_err(|e| BenchError::Source(e.to_string()))?;
    Ok(Some(topo))
}

fn models(args: &BenchArgs) -> Result<Vec<ModelSpec>, BenchError> {
    match &args.source {
        SourceKind::Synthetic { n_nodes, n_edges } => {
            Ok(vec![synthetic(*n_nodes, *n_edges, args.seed)])
        }
        SourceKind::Corpus { path, manifest } => {
            let topo = load_topology(manifest.as_ref())?;
            from_jsonl(path, topo.as_ref()).map_err(|e| BenchError::Source(e.to_string()))
        }
    }
}

/// Beta-rung count matching `sampler_core::build_beta_schedule` (`(num_sweeps /
/// sweeps_per_beta).max(1)`), so the derivation's `num_betas` equals the actual
/// schedule length.
fn num_betas(params: &SampleParams) -> usize {
    (params.num_sweeps / params.sweeps_per_beta.max(1)).max(1)
}

fn bench_one(args: &BenchArgs, spec: &ModelSpec) -> Result<(), BenchError> {
    let params = SampleParams {
        num_reads: args.num_reads,
        num_sweeps: args.num_sweeps,
        sweeps_per_beta: args.sweeps_per_beta,
        beta_range: None,
        seed: args.seed,
    };
    // Warm-up: run without instrumentation so caches/branch predictors settle.
    for _ in 0..args.warmup {
        let _ = sample_ising(&spec.graph, &params, args.algorithm);
    }

    let agg = Arc::new(TimingAggregator::default());
    let folded = args.out_dir.join(format!("{}.folded", spec.model_id));
    let mut measured_ns: u128 = 0;
    run_instrumented(&agg, &folded, || {
        for _ in 0..args.iters {
            let t0 = std::time::Instant::now();
            let _ = sample_ising(&spec.graph, &params, args.algorithm);
            measured_ns += t0.elapsed().as_nanos();
        }
    })?;

    let desc = ModelDescriptor {
        model_id: spec.model_id.clone(),
        n_nodes: spec.graph.h.len(),
        n_edges: spec.graph.edges.len(),
        num_reads: params.num_reads.max(1),
        num_betas: num_betas(&params),
        sweeps_per_beta: params.sweeps_per_beta.max(1),
        seed: params.seed,
    };
    let inputs = ReportInputs {
        backend: "cpu".to_owned(),
        algorithm: algorithm_name(args.algorithm).to_owned(),
        model: desc,
        iters: args.iters,
        warmup: args.warmup,
        measured_model_ns: u64::try_from(measured_ns).unwrap_or(u64::MAX),
    };
    let report = build_report(inputs, &agg.snapshot(), agg.accepts());
    let json =
        serde_json::to_string_pretty(&report).map_err(|e| BenchError::Serialize(e.to_string()))?;
    let json_path = args.out_dir.join(format!("{}.json", spec.model_id));
    std::fs::write(&json_path, json)
        .map_err(|e| BenchError::Io(format!("write report {}: {e}", json_path.display())))?;
    Ok(())
}

/// Run the bench over every model in the source, writing one report + folded
/// file per model into `out_dir`.
pub fn run_bench(args: &BenchArgs) -> Result<(), BenchError> {
    std::fs::create_dir_all(&args.out_dir)
        .map_err(|e| BenchError::Io(format!("create out dir: {e}")))?;
    for spec in models(args)? {
        bench_one(args, &spec)?;
    }
    Ok(())
}

/// A `tracing::Subscriber` that always wants every callsite, with no-op
/// behavior otherwise. See [`ensure_always_interested_global_default`].
struct AlwaysInterested;

impl tracing::Subscriber for AlwaysInterested {
    fn register_callsite(&self, _: &tracing::Metadata<'_>) -> tracing::subscriber::Interest {
        tracing::subscriber::Interest::always()
    }

    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }

    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}

    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}

    fn event(&self, _: &tracing::Event<'_>) {}

    fn enter(&self, _: &tracing::span::Id) {}

    fn exit(&self, _: &tracing::span::Id) {}
}

/// Install [`AlwaysInterested`] as the process-wide global default subscriber,
/// once.
///
/// `tracing`'s per-callsite `Interest` cache is process-global and lazily set
/// on a callsite's first-ever use: whichever thread hits a span macro first
/// asks the *currently active* dispatcher (thread-local `with_default` if set,
/// else the global default) and caches the answer for the rest of the
/// process. With no global default, that fallback is a no-op subscriber that
/// answers `Interest::never()` — so if an unrelated concurrently-running test
/// (no `with_default` of its own) happens to be the first to touch a seam span
/// like `anneal_read`, that span is permanently cached as "never wanted",
/// silently dropping it for every later `with_default` subscriber in the
/// process, including this crate's own bench harness. Setting an
/// always-interested global default closes that gap: every callsite's
/// first-touch interest becomes `Always`, so recording still correctly
/// depends on whichever dispatcher is thread-locally active when the span is
/// actually created. `set_global_default` only succeeds once per process; a
/// later call here is a harmless no-op.
pub(crate) fn ensure_always_interested_global_default() {
    let _ = tracing::subscriber::set_global_default(AlwaysInterested);
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
    ensure_always_interested_global_default();
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
        let params = SampleParams {
            num_reads: 2,
            num_sweeps: 16,
            seed: 1,
            ..Default::default()
        };
        run_instrumented(&agg, &folded, || {
            let _ = sample_ising(&g, &params, Algorithm::Sa);
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
