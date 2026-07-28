//! A `tracing_subscriber::Layer` that accumulates per-span busy-time and counts
//! into a shared aggregator keyed by the span's static name. Single model at a
//! time in bench mode, but the aggregator is `Send + Sync` so it survives the
//! sampler's worker-thread hand-off if ever run through the pump.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use tracing::span::{Attributes, Id};
use tracing::Subscriber;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

/// Accumulated busy-time and close-count for one span name.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PartAccum {
    /// Summed enter→exit busy time in nanoseconds.
    pub total_ns: u128,
    /// Number of times a span with this name closed.
    pub count: u64,
}

/// Thread-safe accumulator: span-name → [`PartAccum`], plus a flip-accept total.
#[derive(Debug, Default)]
pub struct TimingAggregator {
    parts: Mutex<BTreeMap<String, PartAccum>>,
    accepts: AtomicU64,
}

impl TimingAggregator {
    /// Snapshot the current per-part totals (sorted by name).
    #[must_use]
    pub fn snapshot(&self) -> BTreeMap<String, PartAccum> {
        self.parts.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Add `n` to the accepted-flip total.
    pub fn add_accepts(&self, n: u64) {
        self.accepts.fetch_add(n, Ordering::Relaxed);
    }

    /// Total accepted flips recorded so far.
    #[must_use]
    pub fn accepts(&self) -> u64 {
        self.accepts.load(Ordering::Relaxed)
    }

    fn record(&self, name: &str, busy_ns: u128) {
        if let Ok(mut g) = self.parts.lock() {
            let e = g.entry(name.to_owned()).or_default();
            e.total_ns = e.total_ns.saturating_add(busy_ns);
            e.count = e.count.saturating_add(1);
        }
    }
}

/// Per-span open timestamp stored in the span's extensions.
struct SpanStart(Instant);

/// Layer that times spans and folds them into a [`TimingAggregator`].
pub struct TimingLayer {
    agg: std::sync::Arc<TimingAggregator>,
}

impl TimingLayer {
    /// Build a layer feeding `agg`.
    #[must_use]
    pub fn new(agg: std::sync::Arc<TimingAggregator>) -> Self {
        Self { agg }
    }
}

impl<S> Layer<S> for TimingLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, _attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(SpanStart(Instant::now()));
        }
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(&id) {
            let busy = span
                .extensions()
                .get::<SpanStart>()
                .map(|s| s.0.elapsed().as_nanos())
                .unwrap_or(0);
            self.agg.record(span.name(), busy);
        }
    }
}

thread_local! {
    /// Set by the bench harness before a measured `sample_ising` call so the
    /// sampler can push per-read accept counts without a signature change.
    pub static ACTIVE_AGG: std::cell::RefCell<Option<std::sync::Arc<TimingAggregator>>> =
        const { std::cell::RefCell::new(None) };
}

/// Push `n` accepted flips to the thread-active aggregator, if any.
pub fn record_accepts(n: u64) {
    ACTIVE_AGG.with(|a| {
        if let Some(agg) = a.borrow().as_ref() {
            agg.add_accepts(n);
        }
    });
}

/// Install `agg` as the thread-active aggregator for the duration of `f`.
pub fn with_active_agg<R>(agg: &std::sync::Arc<TimingAggregator>, f: impl FnOnce() -> R) -> R {
    ACTIVE_AGG.with(|a| *a.borrow_mut() = Some(std::sync::Arc::clone(agg)));
    let out = f();
    ACTIVE_AGG.with(|a| *a.borrow_mut() = None);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tracing::info_span;
    use tracing_subscriber::prelude::*;

    #[test]
    fn layer_accumulates_named_span_time_and_counts() {
        let agg = Arc::new(TimingAggregator::default());
        let layer = TimingLayer::new(Arc::clone(&agg));
        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            for _ in 0..3 {
                let _s = info_span!("unit_part").entered();
                std::hint::black_box(0u64);
            }
        });
        let snap = agg.snapshot();
        let part = snap.get("unit_part").expect("part recorded");
        assert_eq!(part.count, 3, "three closes expected");
        // total_ns is nonzero busy time; do not assert an exact value.
        assert!(part.total_ns > 0);
    }

    #[test]
    fn accepts_counter_accumulates() {
        let agg = TimingAggregator::default();
        agg.add_accepts(10);
        agg.add_accepts(5);
        assert_eq!(agg.accepts(), 15);
    }
}
