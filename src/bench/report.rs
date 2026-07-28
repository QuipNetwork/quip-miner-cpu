//! Bench report schema (serde) and the derivation math turning a
//! [`crate::bench::timing`] snapshot into per-part timings, derived per-spin and
//! per-flip costs, and a self-consistency residual.
//!
//! The part-record shape (`part`, `scope`, `total_ns`, `count`, `per_call_ns`,
//! `source`) and its `total_ns: u64` / `per_call_ns: f64` types follow the
//! cross-repo reconciliation in `05-reconciliation.md` so this crate's output
//! folds directly into sub-project C's unified per-part record without a
//! translation step. `scope` distinguishes the four non-overlapping top-level
//! seams (`"top_level"`, summed for `residual_ns`) from the `anneal_read`
//! children (`"nested"`, reported for the flame view but excluded from the
//! sum). `source` is always `"tracing"` here — this crate has no nsys/ncu
//! (device) timing source.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::bench::timing::PartAccum;

/// Bump when the JSON shape changes; sub-project C pins on this.
pub const SCHEMA_VERSION: u32 = 1;

/// Non-overlapping wall-time seams; only these are summed for the residual.
/// The `anneal_read` children nest inside it and are reported but not summed.
pub const TOP_LEVEL_PARTS: [&str; 4] = ["cpu_graph_build", "beta_schedule", "anneal_read", "score"];

/// Every span this crate ever times host-side (all `source = "tracing"`).
const SOURCE_TRACING: &str = "tracing";
const SCOPE_TOP_LEVEL: &str = "top_level";
const SCOPE_NESTED: &str = "nested";

/// Static description of the benchmarked model and its sampling knobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDescriptor {
    /// Corpus id or synthetic label.
    pub model_id: String,
    /// Variable count.
    pub n_nodes: usize,
    /// Edge count.
    pub n_edges: usize,
    /// Reads per sample call.
    pub num_reads: usize,
    /// Beta rungs (schedule length).
    pub num_betas: usize,
    /// Sweeps per beta rung.
    pub sweeps_per_beta: usize,
    /// PRNG seed.
    pub seed: u64,
}

/// One part's aggregate and per-call time (05-reconciliation part record).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartTiming {
    /// Span name.
    pub part: String,
    /// `"top_level"` (non-overlapping, summed for `residual_ns`) or `"nested"`
    /// (a child span reported for the flame view but excluded from the sum).
    pub scope: String,
    /// Summed busy-time (ns).
    pub total_ns: u64,
    /// Times the span closed.
    pub count: u64,
    /// `total_ns / count`.
    pub per_call_ns: f64,
    /// Instrumentation source; always `"tracing"` for this crate.
    pub source: String,
}

/// Costs derived by aggregate ÷ frequency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedMetrics {
    /// `num_reads · num_betas · sweeps_per_beta · n_nodes`.
    pub spin_visits: u64,
    /// Accepted flips across all reads.
    pub accepts: u64,
    /// `accepts / spin_visits`.
    pub accept_rate: f64,
    /// `sweep_loop.total_ns / spin_visits`.
    pub per_spin_ns: f64,
    /// `sweep_loop.total_ns / num_reads`.
    pub sweep_loop_ns_per_read: f64,
    /// Per-accepted-flip cost, derived across the corpus (Task 8) by
    /// [`fit_per_flip`]; `None` for a single model.
    pub per_flip_ns: Option<f64>,
}

/// Full bench payload emitted per model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    /// Schema version C pins on.
    pub schema_version: u32,
    /// `"cpu"`.
    pub backend: String,
    /// `"sa"` or `"gibbs"`.
    pub algorithm: String,
    /// Model + knobs.
    pub model: ModelDescriptor,
    /// Measured iterations.
    pub iters: usize,
    /// Warm-up iterations discarded.
    pub warmup: usize,
    /// Wall time of the measured `sample_ising` call(s), summed over `iters`.
    pub measured_model_ns: u64,
    /// Per-part timings (top-level seams + anneal children).
    pub parts: Vec<PartTiming>,
    /// Derived per-spin/per-flip/accept metrics.
    pub derived: DerivedMetrics,
    /// `measured_model_ns − Σ top-level parts`. Signed: normally >= 0, but kept
    /// signed rather than saturated so clock jitter never silently hides a
    /// negative residual (a real bug signal) behind a clamped zero.
    pub residual_ns: i64,
    /// `residual_ns / measured_model_ns`.
    pub residual_frac: f64,
}

/// Convert a `u128` nanosecond duration to `u64`, saturating at `u64::MAX`. A
/// single model's total is ~1e9-1e10 ns, far below `u64::MAX` (~1.8e19); the
/// saturation only guards against a pathological run and never panics.
fn saturating_u64(v: u128) -> u64 {
    u64::try_from(v).unwrap_or(u64::MAX)
}

/// Compute derived metrics from the model shape and a timing snapshot.
///
/// `iters` is the number of measured `sample_ising` calls folded into `snap`
/// and `accepts` (a bench run repeats the model `iters` times and the
/// aggregator sums across all of them, so the visit count must scale by
/// `iters` too or `accept_rate` could read above 1.0). Pass `1` for a
/// single-call snapshot (e.g. the Task 4 unit tests below).
#[must_use]
pub fn derive_metrics(
    desc: &ModelDescriptor,
    snap: &BTreeMap<String, PartAccum>,
    accepts: u64,
    iters: usize,
) -> DerivedMetrics {
    let iters = iters.max(1) as u64;
    let reads_total = (desc.num_reads as u64).saturating_mul(iters);
    let spin_visits = reads_total
        .saturating_mul(desc.num_betas as u64)
        .saturating_mul(desc.sweeps_per_beta as u64)
        .saturating_mul(desc.n_nodes as u64);
    let sweep_total = snap.get("sweep_loop").map_or(0u128, |p| p.total_ns);
    let per_spin_ns = if spin_visits > 0 {
        sweep_total as f64 / spin_visits as f64
    } else {
        0.0
    };
    let accept_rate = if spin_visits > 0 {
        accepts as f64 / spin_visits as f64
    } else {
        0.0
    };
    let sweep_loop_ns_per_read = if reads_total > 0 {
        sweep_total as f64 / reads_total as f64
    } else {
        0.0
    };
    DerivedMetrics {
        spin_visits,
        accepts,
        accept_rate,
        per_spin_ns,
        sweep_loop_ns_per_read,
        per_flip_ns: None,
    }
}

/// Residual `(ns, fraction)` from measured time minus summed top-level parts.
#[must_use]
pub fn residual(measured_ns: u64, parts: &[PartTiming], top_level: &[&str]) -> (i64, f64) {
    let summed: u64 = parts
        .iter()
        .filter(|p| top_level.contains(&p.part.as_str()))
        .map(|p| p.total_ns)
        .sum();
    let res =
        i64::try_from(measured_ns).unwrap_or(i64::MAX) - i64::try_from(summed).unwrap_or(i64::MAX);
    let frac = if measured_ns > 0 {
        res as f64 / measured_ns as f64
    } else {
        0.0
    };
    (res, frac)
}

/// Fixed inputs for [`build_report`], grouped to stay within the 5-positional-
/// parameter limit (the timing snapshot and accept count vary per call and
/// stay as separate arguments).
#[derive(Debug, Clone)]
pub struct ReportInputs {
    /// `"cpu"`.
    pub backend: String,
    /// `"sa"` or `"gibbs"`.
    pub algorithm: String,
    /// Model + knobs.
    pub model: ModelDescriptor,
    /// Measured iterations.
    pub iters: usize,
    /// Warm-up iterations discarded.
    pub warmup: usize,
    /// Wall time of the measured `sample_ising` call(s), summed over `iters`.
    pub measured_model_ns: u64,
}

/// Assemble a [`BenchReport`] from a snapshot.
#[must_use]
pub fn build_report(
    inputs: ReportInputs,
    snap: &BTreeMap<String, PartAccum>,
    accepts: u64,
) -> BenchReport {
    let ReportInputs {
        backend,
        algorithm,
        model,
        iters,
        warmup,
        measured_model_ns,
    } = inputs;
    let parts: Vec<PartTiming> = snap
        .iter()
        .map(|(name, a)| {
            let total_ns = saturating_u64(a.total_ns);
            let scope = if TOP_LEVEL_PARTS.contains(&name.as_str()) {
                SCOPE_TOP_LEVEL
            } else {
                SCOPE_NESTED
            };
            PartTiming {
                part: name.clone(),
                scope: scope.to_owned(),
                total_ns,
                count: a.count,
                per_call_ns: if a.count > 0 {
                    total_ns as f64 / a.count as f64
                } else {
                    0.0
                },
                source: SOURCE_TRACING.to_owned(),
            }
        })
        .collect();
    let derived = derive_metrics(&model, snap, accepts, iters);
    let (residual_ns, residual_frac) = residual(measured_model_ns, &parts, &TOP_LEVEL_PARTS);
    BenchReport {
        schema_version: SCHEMA_VERSION,
        backend,
        algorithm,
        model,
        iters,
        warmup,
        measured_model_ns,
        parts,
        derived,
        residual_ns,
        residual_frac,
    }
}

/// Fit `ns = a·visits + b·accepts` by least squares over corpus rows
/// `(visits_per_read, accepts_per_read, sweep_loop_ns_per_read)` and return the
/// per-flip slope `b`. `None` if the normal-equations matrix is singular
/// (regressors collinear — insufficient accept-rate spread across the corpus).
///
/// This is a corpus-level fit, populated by the driver (sub-project C) after
/// collecting per-model rows from [`DerivedMetrics`] — a single model leaves
/// `per_flip_ns` `None`.
#[must_use]
pub fn fit_per_flip(rows: &[(f64, f64, f64)]) -> Option<f64> {
    // Normal equations for [a, b]: [[Svv, Sva],[Sva, Saa]] [a,b]^T = [Svy, Say].
    let (mut svv, mut sva, mut saa, mut svy, mut say) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &(v, acc, y) in rows {
        svv += v * v;
        sva += v * acc;
        saa += acc * acc;
        svy += v * y;
        say += acc * y;
    }
    let det = svv * saa - sva * sva;
    if det.abs() < 1e-9 {
        return None;
    }
    // b = (Svv·Say − Sva·Svy) / det.
    Some((svv * say - sva * svy) / det)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn accum(total_ns: u128, count: u64) -> PartAccum {
        PartAccum { total_ns, count }
    }

    #[test]
    fn per_call_and_per_spin_derive_from_frequency() {
        // 2 reads, 4 betas, 3 sweeps_per_beta, 5 nodes → 120 visits.
        let desc = ModelDescriptor {
            model_id: "m".into(),
            n_nodes: 5,
            n_edges: 6,
            num_reads: 2,
            num_betas: 4,
            sweeps_per_beta: 3,
            seed: 1,
        };
        let mut snap = BTreeMap::new();
        snap.insert("sweep_loop".to_owned(), accum(1_200_000, 2)); // 1.2 ms over 2 reads
        snap.insert("anneal_read".to_owned(), accum(1_400_000, 2));
        let derived = derive_metrics(&desc, &snap, /* accepts= */ 60, /* iters= */ 1);
        assert_eq!(derived.spin_visits, 120);
        // per_spin = sweep_loop_total / visits = 1_200_000 / 120 = 10_000 ns.
        assert!((derived.per_spin_ns - 10_000.0).abs() < 1e-6);
        // accept_rate = 60/120 = 0.5.
        assert!((derived.accept_rate - 0.5).abs() < 1e-12);
        // sweep_loop_ns_per_read = 600_000.
        assert!((derived.sweep_loop_ns_per_read - 600_000.0).abs() < 1e-6);
        assert!(derived.per_flip_ns.is_none());
    }

    #[test]
    fn spin_visits_scale_with_iters_so_accept_rate_stays_bounded() {
        // A bench run repeats one read/beta/sweep/node model 3 times; the
        // aggregator's snapshot and accepts sum over all 3 measured calls, so
        // spin_visits must scale by iters or accept_rate would exceed 1.0.
        let desc = ModelDescriptor {
            model_id: "m".into(),
            n_nodes: 4,
            n_edges: 3,
            num_reads: 2,
            num_betas: 2,
            sweeps_per_beta: 1,
            seed: 0,
        };
        let iters = 3;
        let single_call_visits = 2u64 * 2 * 4; // reads * betas * sweeps_per_beta(1, elided) * nodes = 16
        let mut snap = BTreeMap::new();
        snap.insert("sweep_loop".to_owned(), accum(1_000, (iters as u64) * 2));
        // Max possible accepts across 3 calls: iters * single_call_visits.
        let accepts = (iters as u64) * single_call_visits;
        let derived = derive_metrics(&desc, &snap, accepts, iters);
        assert_eq!(derived.spin_visits, (iters as u64) * single_call_visits);
        assert!((derived.accept_rate - 1.0).abs() < 1e-12);
    }

    fn part(name: &str, total_ns: u64) -> PartTiming {
        PartTiming {
            part: name.to_owned(),
            scope: if TOP_LEVEL_PARTS.contains(&name) {
                SCOPE_TOP_LEVEL.to_owned()
            } else {
                SCOPE_NESTED.to_owned()
            },
            total_ns,
            count: 1,
            per_call_ns: total_ns as f64,
            source: SOURCE_TRACING.to_owned(),
        }
    }

    #[test]
    fn residual_is_measured_minus_summed_parts() {
        let parts = vec![part("cpu_graph_build", 300), part("beta_schedule", 400)];
        let (res_ns, res_frac) = residual(1000, &parts, &["cpu_graph_build", "beta_schedule"]);
        assert_eq!(res_ns, 300); // 1000 - 700
        assert!((res_frac - 0.3).abs() < 1e-12);
    }

    #[test]
    fn part_record_carries_scope_and_source() {
        let top = part("cpu_graph_build", 10);
        assert_eq!(top.scope, "top_level");
        assert_eq!(top.source, "tracing");
        let nested = part("sweep_loop", 10);
        assert_eq!(nested.scope, "nested");
    }

    #[test]
    fn report_serializes_stable_schema() {
        let desc = ModelDescriptor {
            model_id: "m".into(),
            n_nodes: 2,
            n_edges: 1,
            num_reads: 1,
            num_betas: 1,
            sweeps_per_beta: 1,
            seed: 0,
        };
        let mut snap = BTreeMap::new();
        snap.insert("anneal_read".to_owned(), accum(500, 1));
        snap.insert("sweep_loop".to_owned(), accum(400, 1));
        snap.insert("score".to_owned(), accum(50, 1));
        let inputs = ReportInputs {
            backend: "cpu".to_owned(),
            algorithm: "sa".to_owned(),
            model: desc,
            iters: 5,
            warmup: 1,
            measured_model_ns: 600,
        };
        let rep = build_report(inputs, &snap, 0);
        let json = serde_json::to_string(&rep).expect("serialize");
        assert!(json.contains("\"schema_version\""));
        assert!(json.contains("\"parts\""));
        assert!(json.contains("\"scope\""));
        assert!(json.contains("\"source\""));
        assert!(json.contains("\"per_spin_ns\""));
        assert!(json.contains("\"residual_ns\""));
    }

    #[test]
    fn fit_per_flip_recovers_slope() {
        // Model: ns = a*visits + b*accepts, a=10, b=40. Build 4 rows.
        let a = 10.0;
        let b = 40.0;
        let rows: Vec<(f64, f64, f64)> =
            [(100.0, 20.0), (100.0, 60.0), (200.0, 40.0), (200.0, 120.0)]
                .iter()
                .map(|&(v, acc)| (v, acc, a * v + b * acc))
                .collect();
        let per_flip = fit_per_flip(&rows).expect("fit");
        assert!(
            (per_flip - 40.0).abs() < 1e-6,
            "recovered per_flip = {per_flip}"
        );
    }

    #[test]
    fn fit_per_flip_needs_variation() {
        // All rows collinear in (visits, accepts) → no unique fit.
        let rows = vec![(100.0, 50.0, 1000.0), (200.0, 100.0, 2000.0)];
        assert!(fit_per_flip(&rows).is_none());
    }
}
