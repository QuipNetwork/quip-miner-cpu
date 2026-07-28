# quip-miner-cpu

CPU Ising miners for the [quip.network](https://gitlab.com/quip.network) v0.3
mining protocol: simulated annealing (`quip-cpu-sa`) and single-site chromatic
Gibbs (`quip-cpu-gibbs`), shipped as separate binaries.

The sampler runs one model per core (model-level parallelism); each model's
reads are sequential and cache-local. Energies are scored with the canonical
`quip_protocol::scoring::energy_milli` so results match consensus.

## Binaries

| binary | algorithm |
|--------|-----------|
| `quip-cpu-sa` | simulated annealing (Metropolis) |
| `quip-cpu-gibbs` | single-site heat-bath Gibbs |
| `quip-cpu-bench` | per-part timing harness (see [Benchmarking](#benchmarking)) |

Prebuilt `amd64`/`arm64` binaries are attached to each
[Release](https://gitlab.com/quip.network/quip-miner-cpu/-/releases).

## Build

```sh
cargo build --release        # needs protoc on PATH (protobuf-compiler)
```

Shared protocol crates (`quip-proto`, `quip-protocol`, `quip-miner-core`) are
git dependencies pinned to a `shared-vX.Y.Z` tag of `quip-protocol`.

## Running

**Connect to a coordinator** (production):

```sh
quip-cpu-sa --quip-coordinator unix:///run/quip/coord.sock
```

**Driver / fixed-input (run in isolation, no chain).** Use the coordinator's
`drive` harness pointed at the binary — `--source random` for golden-drawn
problems, `--source list <jsonl>` for a fixed replay:

```sh
quip-coordinator drive --miner ./quip-cpu-sa \
  --source random --topology-preset advantage2-system1 \
  --count 8 --num-reads 16 --num-sweeps 1030 --report out.jsonl
```

**Introspection:**

```sh
quip-cpu-sa --capabilities   # capabilities JSON
quip-cpu-sa --check          # probe the backend is runnable
```

## Benchmarking

`quip-cpu-bench` runs a model through the sampler under a `tracing` subscriber
and emits per-part timing plus a flame graph, so the sampler's time budget can
be attributed to each annealing seam instead of only the whole-model total.

```sh
quip-cpu-bench --nodes 512 --edges 2048 --iters 5 --out-dir bench-out
# or against a coordinator corpus (nonce-refs need --topology for the topology):
quip-cpu-bench --source instances.jsonl --topology topology.spec.json --out-dir bench-out
```

`--source` is the coordinator's `instances.jsonl` (one JSON object per line,
keyed on `nonce`; unrelated keys ride along and are ignored). `--topology` is
the coordinator's `topology.spec.json`
(`{nodes, edges, allowed_h_milli, allowed_j_milli}`); each nonce-ref entry is
redrawn against it via `quip_protocol::chacha8::draw_ising_milli`. `--limit K`
benches only the first `K` corpus models.

Flags: `--algorithm sa|gibbs`, `--nodes`/`--edges` (synthetic model) or
`--source`/`--topology`/`--limit` (corpus JSONL), `--num-reads`,
`--num-sweeps`, `--sweeps-per-beta`, `--seed`, `--warmup`, `--iters`,
`--out-dir`.

Each model writes `<out_dir>/<model_id>.json` and `<out_dir>/<model_id>.folded`:

- `schema_version` — bump signals a JSON shape change.
- `parts[]` — one entry per traced span (`part`, `scope`, `total_ns`, `count`,
  `per_call_ns`, `source`). `scope` is `"top_level"` for the four
  non-overlapping seams (`cpu_graph_build`, `beta_schedule`, `anneal_read`,
  `score`) summed into `residual_ns`, or `"nested"` for `anneal_read`'s
  children (`random_spins`, `seed_heff`, `sweep_loop`), reported for the flame
  view but excluded from that sum.
- `derived` — `per_spin_ns` and `accept_rate` computed as aggregate ÷
  frequency (`sweep_loop.total_ns` ÷ spin visits), plus `sweep_loop_ns_per_read`.
- `residual_ns`/`residual_frac` — `measured_model_ns` minus the summed
  top-level parts; a large fraction signals a missing seam.
- `<model_id>.folded` — `tracing-flame` folded stacks; render with
  `inferno-flamegraph < model.folded > model.svg`.

The headline JSON uses coarse, always-on seam spans (negligible overhead —
entered O(reads) times per model). `--features fine-spans` also spans every
spin decision and accepted flip inside the hot loop, for a diagnostic
cross-check. That build measurably distorts absolute timing, so headline
numbers never use it. Optional external cross-check:
`cargo flamegraph --bin quip-cpu-bench -- --nodes 512 --edges 2048`.

## Tests

```sh
cargo test --release
```

Conformance/golden and handshake tests drive the binary in isolation via
`quip-mock-coordinator` and check energies against `conformance/golden_vectors.json`.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
