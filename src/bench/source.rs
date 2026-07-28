//! Bench model sources: synthetic (flag-driven) and corpus JSONL.
//!
//! Corpus entries are auto-detected: an entry with `nonce` is redrawn against a
//! [`Topology`] via [`quip_protocol::chacha8::draw_ising_milli`]; an entry with
//! `h_milli` is used verbatim. This mirrors sub-project A's `hardest_models`
//! nonce-ref format so the same `instances.jsonl` feeds CPU and CUDA benches.

use std::path::Path;

use quip_protocol::chacha8::draw_ising_milli;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;

use crate::IsingGraph;

/// Topology shape for redrawing nonce-ref entries (from A's `manifest.json`).
#[derive(Debug, Clone)]
pub struct Topology {
    /// Variable count.
    pub n_nodes: usize,
    /// Edge count.
    pub n_edges: usize,
    /// Allowed field milli-values (ChaCha8 draw set).
    pub allowed_h_milli: Vec<i32>,
    /// Allowed coupling milli-values.
    pub allowed_j_milli: Vec<i32>,
    /// Fixed edge list for this topology.
    pub edges: Vec<(usize, usize)>,
}

/// A ready-to-run model plus its id.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// Corpus id or synthetic label.
    pub model_id: String,
    /// Built graph.
    pub graph: IsingGraph,
}

/// Model-source failure.
#[derive(Debug)]
pub enum SourceError {
    /// File read failure.
    Io(String),
    /// JSON decode failure at 1-based `line`.
    Parse { line: usize, reason: String },
    /// Entry has neither `nonce` nor `h_milli`, or both.
    Ambiguous { line: usize },
    /// Nonce-ref entry but no `--manifest` topology supplied.
    MissingTopology { line: usize },
    /// Nonce hex not 32 bytes.
    BadNonce { line: usize },
    /// `draw_ising_milli` rejected the topology.
    Draw { line: usize, reason: String },
}

impl std::fmt::Display for SourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(m) => write!(f, "read corpus: {m}"),
            Self::Parse { line, reason } => write!(f, "line {line}: {reason}"),
            Self::Ambiguous { line } => {
                write!(f, "line {line}: entry needs exactly one of `nonce` or `h_milli`")
            }
            Self::MissingTopology { line } => {
                write!(f, "line {line}: nonce-ref entry requires --manifest topology")
            }
            Self::BadNonce { line } => write!(f, "line {line}: nonce must be 32 bytes hex"),
            Self::Draw { line, reason } => write!(f, "line {line}: draw failed: {reason}"),
        }
    }
}

impl std::error::Error for SourceError {}

#[derive(Debug, Deserialize)]
struct EntryJson {
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    nonce: Option<String>,
    #[serde(default)]
    h_milli: Option<Vec<i32>>,
    #[serde(default)]
    j_milli: Option<Vec<i32>>,
    #[serde(default)]
    edges: Option<Vec<(usize, usize)>>,
}

fn milli_to_unit(v: &[i32]) -> Vec<f64> {
    v.iter().map(|&m| f64::from(m) / 1000.0).collect()
}

fn parse_nonce(hex: &str, line: usize) -> Result<[u8; 32], SourceError> {
    if hex.len() != 64 {
        return Err(SourceError::BadNonce { line });
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        let s = hex.get(i * 2..i * 2 + 2).ok_or(SourceError::BadNonce { line })?;
        *byte = u8::from_str_radix(s, 16).map_err(|_| SourceError::BadNonce { line })?;
    }
    Ok(out)
}

/// Build a synthetic model: random ±/0 fields and a random edge subset.
#[must_use]
pub fn synthetic(n_nodes: usize, n_edges: usize, seed: u64) -> ModelSpec {
    let mut rng = SmallRng::seed_from_u64(seed);
    let h: Vec<f64> = (0..n_nodes)
        .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
        .collect();
    let mut edges = Vec::with_capacity(n_edges);
    let mut j = Vec::with_capacity(n_edges);
    let mut guard = 0usize;
    while edges.len() < n_edges && n_nodes >= 2 && guard < n_edges * 8 {
        let u = rng.gen_range(0..n_nodes);
        let v = rng.gen_range(0..n_nodes);
        guard += 1;
        if u == v {
            continue;
        }
        edges.push((u, v));
        j.push(if rng.gen::<bool>() { 1.0 } else { -1.0 });
    }
    ModelSpec {
        model_id: format!("synthetic-n{n_nodes}-e{n_edges}-s{seed}"),
        graph: IsingGraph::new(h, j, edges),
    }
}

/// Parse a JSONL corpus into models; nonce-refs redraw against `topology`.
pub fn from_jsonl(path: &Path, topology: Option<&Topology>) -> Result<Vec<ModelSpec>, SourceError> {
    let text = std::fs::read_to_string(path).map_err(|e| SourceError::Io(e.to_string()))?;
    let mut out = Vec::new();
    for (idx, raw) in text.lines().enumerate() {
        let line = idx + 1;
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let e: EntryJson =
            serde_json::from_str(raw).map_err(|err| SourceError::Parse { line, reason: err.to_string() })?;
        let has_nonce = e.nonce.is_some();
        let has_explicit = e.h_milli.is_some();
        let spec = match (has_nonce, has_explicit) {
            (true, false) => build_nonce(&e, topology, line)?,
            (false, true) => build_explicit(&e, line),
            _ => return Err(SourceError::Ambiguous { line }),
        };
        out.push(spec);
    }
    Ok(out)
}

fn build_explicit(e: &EntryJson, line: usize) -> ModelSpec {
    let h = milli_to_unit(e.h_milli.as_deref().unwrap_or(&[]));
    let j = milli_to_unit(e.j_milli.as_deref().unwrap_or(&[]));
    let edges = e.edges.clone().unwrap_or_default();
    ModelSpec {
        model_id: e.model_id.clone().unwrap_or_else(|| format!("explicit-{line}")),
        graph: IsingGraph::new(h, j, edges),
    }
}

fn build_nonce(e: &EntryJson, topology: Option<&Topology>, line: usize) -> Result<ModelSpec, SourceError> {
    let topo = topology.ok_or(SourceError::MissingTopology { line })?;
    let nonce = parse_nonce(e.nonce.as_deref().unwrap_or(""), line)?;
    let (h_milli, j_milli) = draw_ising_milli(
        nonce,
        topo.n_nodes,
        topo.n_edges,
        &topo.allowed_h_milli,
        &topo.allowed_j_milli,
    )
    .map_err(|err| SourceError::Draw { line, reason: err.to_string() })?;
    Ok(ModelSpec {
        model_id: e.model_id.clone().unwrap_or_else(|| format!("nonce-{line}")),
        graph: IsingGraph::new(milli_to_unit(&h_milli), milli_to_unit(&j_milli), topo.edges.clone()),
    })
}

#[derive(Debug, Deserialize)]
struct ManifestJson {
    n_nodes: usize,
    n_edges: usize,
    allowed_h_milli: Vec<i32>,
    allowed_j_milli: Vec<i32>,
    edges: Vec<(usize, usize)>,
}

/// Parse A's `manifest.json` into a [`Topology`] for nonce redraws.
///
/// **Open coupling (keep-and-flag):** the exact field names are owned by
/// sub-project A (`10-plan-coordinator-download.md`). If A's emitted schema
/// differs (e.g. `topology.allowed_h`), adjust [`ManifestJson`]'s field names
/// / add `#[serde(rename = "...")]` to match once A lands.
pub fn topology_from_manifest_json(text: &str) -> Result<Topology, SourceError> {
    let m: ManifestJson =
        serde_json::from_str(text).map_err(|e| SourceError::Parse { line: 0, reason: e.to_string() })?;
    Ok(Topology {
        n_nodes: m.n_nodes,
        n_edges: m.n_edges,
        allowed_h_milli: m.allowed_h_milli,
        allowed_j_milli: m.allowed_j_milli,
        edges: m.edges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    #[test]
    fn synthetic_has_requested_shape() {
        let m = synthetic(8, 12, 42);
        assert_eq!(m.graph.h.len(), 8);
        assert_eq!(m.graph.edges.len(), m.graph.j.len());
        assert!(m.graph.edges.len() <= 12);
        assert!(m.graph.edges.iter().all(|&(u, v)| u < 8 && v < 8 && u != v));
    }

    #[test]
    fn explicit_entry_parses_verbatim() {
        let dir = std::env::temp_dir().join(format!("quipsrc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("explicit.jsonl");
        let mut f = std::fs::File::create(&p).unwrap();
        writeln!(f, r#"{{"model_id":"x","h_milli":[500,-500],"j_milli":[-1000],"edges":[[0,1]]}}"#).unwrap();
        let specs = from_jsonl(&p, None).expect("parse");
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].model_id, "x");
        assert_eq!(specs[0].graph.h, vec![0.5, -0.5]); // milli → unit
        assert_eq!(specs[0].graph.j, vec![-1.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn nonce_entry_redraws_via_topology() {
        let dir = std::env::temp_dir().join(format!("quipnonce-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("nonce.jsonl");
        let mut f = std::fs::File::create(&p).unwrap();
        // 32-byte nonce hex (all zeros).
        writeln!(f, r#"{{"model_id":"n0","nonce":"{}"}}"#, "00".repeat(32)).unwrap();
        let topo = Topology {
            n_nodes: 2,
            n_edges: 1,
            allowed_h_milli: vec![-1000, 1000],
            allowed_j_milli: vec![-1000, 1000],
            edges: vec![(0, 1)],
        };
        let specs = from_jsonl(&p, Some(&topo)).expect("redraw");
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].graph.h.len(), 2);
        assert_eq!(specs[0].graph.edges, vec![(0, 1)]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn nonce_without_topology_errors() {
        let dir = std::env::temp_dir().join(format!("quipnotopo-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("nonce.jsonl");
        std::fs::write(&p, format!("{{\"nonce\":\"{}\"}}\n", "00".repeat(32))).unwrap();
        let err = from_jsonl(&p, None).unwrap_err();
        assert!(matches!(err, SourceError::MissingTopology { .. }));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn manifest_json_round_trips_into_topology() {
        let json = r#"{
            "n_nodes": 4,
            "n_edges": 3,
            "allowed_h_milli": [-1000, 1000],
            "allowed_j_milli": [-1000, 1000],
            "edges": [[0,1],[1,2],[2,3]]
        }"#;
        let topo = topology_from_manifest_json(json).expect("parse manifest");
        assert_eq!(topo.n_nodes, 4);
        assert_eq!(topo.edges, vec![(0, 1), (1, 2), (2, 3)]);
    }
}
