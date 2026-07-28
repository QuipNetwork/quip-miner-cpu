//! Bench model sources: synthetic (flag-driven) and corpus JSONL.
//!
//! Corpus entries are auto-detected: an entry with `nonce` is redrawn against a
//! [`Topology`] (parsed from a coordinator `topology.spec.json` via
//! [`parse_topology_spec`]) using [`quip_protocol::chacha8::draw_ising_milli`];
//! an entry with `h_milli` is used verbatim. The corpus JSONL is the
//! coordinator's `instances.jsonl`: one JSON object per line keyed on `nonce`
//! (un-prefixed 64-char hex = 32 bytes), with unrelated keys
//! (`topology_hash`, `energy_milli`, `salt_hex`, `qblock_id`, …) ignored.

use std::collections::HashMap;
use std::path::Path;

use quip_protocol::chacha8::draw_ising_milli;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;

use crate::IsingGraph;

/// Topology shape for redrawing nonce-ref entries, resolved from a
/// `topology.spec.json` file (see [`parse_topology_spec`]).
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
    /// Edge list, native node ids resolved to dense `0..n_nodes` positions
    /// (matching `IsingGraph`'s index convention).
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
    /// Nonce-ref entry but no `--topology` spec supplied.
    MissingTopology { line: usize },
    /// Nonce hex not 32 bytes.
    BadNonce { line: usize },
    /// `draw_ising_milli` rejected the topology.
    Draw { line: usize, reason: String },
    /// A topology-spec `edges` endpoint is not a known node id.
    TopologyEdgeUnknownNode { edge_index: usize, node: u32 },
}

impl std::fmt::Display for SourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(m) => write!(f, "read corpus: {m}"),
            Self::Parse { line, reason } => write!(f, "line {line}: {reason}"),
            Self::Ambiguous { line } => {
                write!(
                    f,
                    "line {line}: entry needs exactly one of `nonce` or `h_milli`"
                )
            }
            Self::MissingTopology { line } => {
                write!(
                    f,
                    "line {line}: nonce-ref entry requires --topology topology spec"
                )
            }
            Self::BadNonce { line } => write!(f, "line {line}: nonce must be 32 bytes hex"),
            Self::Draw { line, reason } => write!(f, "line {line}: draw failed: {reason}"),
            Self::TopologyEdgeUnknownNode { edge_index, node } => write!(
                f,
                "topology spec: edge {edge_index} references unknown node id {node}"
            ),
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
        let s = hex
            .get(i * 2..i * 2 + 2)
            .ok_or(SourceError::BadNonce { line })?;
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
        let e: EntryJson = serde_json::from_str(raw).map_err(|err| SourceError::Parse {
            line,
            reason: err.to_string(),
        })?;
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
        model_id: e
            .model_id
            .clone()
            .unwrap_or_else(|| format!("explicit-{line}")),
        graph: IsingGraph::new(h, j, edges),
    }
}

fn build_nonce(
    e: &EntryJson,
    topology: Option<&Topology>,
    line: usize,
) -> Result<ModelSpec, SourceError> {
    let topo = topology.ok_or(SourceError::MissingTopology { line })?;
    let nonce = parse_nonce(e.nonce.as_deref().unwrap_or(""), line)?;
    let (h_milli, j_milli) = draw_ising_milli(
        nonce,
        topo.n_nodes,
        topo.n_edges,
        &topo.allowed_h_milli,
        &topo.allowed_j_milli,
    )
    .map_err(|err| SourceError::Draw {
        line,
        reason: err.to_string(),
    })?;
    Ok(ModelSpec {
        model_id: e
            .model_id
            .clone()
            .unwrap_or_else(|| format!("nonce-{line}")),
        graph: IsingGraph::new(
            milli_to_unit(&h_milli),
            milli_to_unit(&j_milli),
            topo.edges.clone(),
        ),
    })
}

/// Coordinator `topology.spec.json` shape: `nodes` are native (possibly
/// sparse) ids in received order; `edges` reference those ids, not positions.
#[derive(Debug, Deserialize)]
struct TopologySpecJson {
    nodes: Vec<u32>,
    #[serde(default)]
    edges: Vec<(u32, u32)>,
    allowed_h_milli: Vec<i32>,
    allowed_j_milli: Vec<i32>,
}

/// Parse a coordinator `topology.spec.json` document into a [`Topology`] for
/// nonce redraws, resolving `edges`' native node ids to dense positions (id →
/// index in `nodes`' received order), matching the coordinator's own
/// `TopologyCache` resolution so sparse ids (e.g. D-Wave qubits) work.
///
/// # Errors
///
/// Returns [`SourceError::Parse`] on invalid JSON or
/// [`SourceError::TopologyEdgeUnknownNode`] if an edge references a node id
/// absent from `nodes`.
pub fn parse_topology_spec(text: &str) -> Result<Topology, SourceError> {
    let spec: TopologySpecJson = serde_json::from_str(text).map_err(|e| SourceError::Parse {
        line: 0,
        reason: e.to_string(),
    })?;
    let pos: HashMap<u32, usize> = spec
        .nodes
        .iter()
        .enumerate()
        .map(|(i, &node)| (node, i))
        .collect();
    let mut edges = Vec::with_capacity(spec.edges.len());
    for (edge_index, &(u, v)) in spec.edges.iter().enumerate() {
        let pu = *pos.get(&u).ok_or(SourceError::TopologyEdgeUnknownNode {
            edge_index,
            node: u,
        })?;
        let pv = *pos.get(&v).ok_or(SourceError::TopologyEdgeUnknownNode {
            edge_index,
            node: v,
        })?;
        edges.push((pu, pv));
    }
    Ok(Topology {
        n_nodes: spec.nodes.len(),
        n_edges: edges.len(),
        allowed_h_milli: spec.allowed_h_milli,
        allowed_j_milli: spec.allowed_j_milli,
        edges,
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
        writeln!(
            f,
            r#"{{"model_id":"x","h_milli":[500,-500],"j_milli":[-1000],"edges":[[0,1]]}}"#
        )
        .unwrap();
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
    fn topology_spec_json_round_trips_into_topology() {
        let json = r#"{
            "nodes": [0, 1, 2, 3],
            "edges": [[0,1],[1,2],[2,3]],
            "allowed_h_milli": [-1000, 1000],
            "allowed_j_milli": [-1000, 1000]
        }"#;
        let topo = parse_topology_spec(json).expect("parse topology spec");
        assert_eq!(topo.n_nodes, 4);
        assert_eq!(topo.n_edges, 3);
        assert_eq!(topo.edges, vec![(0, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn topology_spec_resolves_sparse_native_node_ids_to_positions() {
        // D-Wave-style sparse ids: edges reference ids, not array positions.
        let json = r#"{
            "nodes": [0, 12, 2400],
            "edges": [[0, 12], [12, 2400]],
            "allowed_h_milli": [-1000, 0, 1000],
            "allowed_j_milli": [-1000, 1000]
        }"#;
        let topo = parse_topology_spec(json).expect("parse topology spec");
        assert_eq!(topo.n_nodes, 3);
        // Ids 0, 12, 2400 map to positions 0, 1, 2 in received order.
        assert_eq!(topo.edges, vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn topology_spec_rejects_edge_with_unknown_node_id() {
        let json = r#"{
            "nodes": [0, 1],
            "edges": [[0, 5]],
            "allowed_h_milli": [1000],
            "allowed_j_milli": [1000]
        }"#;
        let err = parse_topology_spec(json).unwrap_err();
        assert!(matches!(
            err,
            SourceError::TopologyEdgeUnknownNode {
                edge_index: 0,
                node: 5
            }
        ));
    }

    #[test]
    fn topology_spec_extra_keys_are_ignored() {
        // Real instances.jsonl lines carry extra keys (topology_hash,
        // energy_milli, ...); the corpus EntryJson must ignore them too.
        let dir = std::env::temp_dir().join(format!("quipextra-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("nonce.jsonl");
        writeln!(
            std::fs::File::create(&p).unwrap(),
            r#"{{"nonce":"{}","topology_hash":"deadbeef","energy_milli":-1000,"salt_hex":"aa","qblock_id":1}}"#,
            "00".repeat(32)
        )
        .unwrap();
        let topo = Topology {
            n_nodes: 2,
            n_edges: 1,
            allowed_h_milli: vec![-1000, 1000],
            allowed_j_milli: vec![-1000, 1000],
            edges: vec![(0, 1)],
        };
        let specs = from_jsonl(&p, Some(&topo)).expect("redraw ignoring extra keys");
        assert_eq!(specs.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
