# quip-solver-cpu

CPU Ising miners for the [quip.network](https://gitlab.com/quip.network) v0.3
mining protocol: simulated annealing (`quip-cpu-sa`), single-site chromatic
Gibbs (`quip-cpu-gibbs`), and discrete Simulated Bifurcation (`quip-cpu-sb`),
shipped as separate binaries.

The sampler runs one model per core (model-level parallelism); each model's
reads are sequential and cache-local. Energies are scored with the canonical
`quip_protocol::scoring::energy_milli` so results match consensus.

## Binaries

| binary | algorithm | track |
|--------|-----------|-------|
| `quip-cpu-sa` | simulated annealing (Metropolis) | production |
| `quip-cpu-gibbs` | single-site heat-bath Gibbs | production |
| `quip-cpu-sb` | discrete Simulated Bifurcation | production |
| `quip-cpu-bench` | per-part timing of the scalar SA kernel (see [Benchmarking](#benchmarking)) | production |
| `quip-cpu-bsb` | ballistic Simulated Bifurcation | experimental |
| `quip-cpu-hdsb` | heated discrete Simulated Bifurcation | experimental |
| `quip-cpu-hbsb` | heated ballistic Simulated Bifurcation | experimental |
| `quip-cpu-gbsb` | generalized ballistic Simulated Bifurcation, edge-of-chaos control | experimental |
| `quip-cpu-gdsb` | edge-of-chaos control on the discrete coupling | experimental |
| `quip-cpu-tedsb` | tabu-enhanced discrete Simulated Bifurcation | experimental |
| `quip-cpu-sbqa` | replica-ring discrete Simulated Bifurcation | experimental |
| `quip-cpu-ggdsb` | globally guided discrete Simulated Bifurcation | experimental |
| `quip-cpu-fsa` | simulated annealing with tabulated Metropolis thresholds | experimental |
| `quip-cpu-msa` | multi-spin coded simulated annealing, 64 reads per word | experimental |
| `quip-cpu-mps` | tensor network: imaginary-time TEBD with exact sampling | experimental |
| `quip-cpu-mfa` | mean-field annealing (the same kernel at bond dimension 1) | experimental |
| `quip-cpu-flatiron` | belief-propagation tensor network on the problem graph | experimental |

The MSA miner uses color-ordered sweeps with 64 replicas per word. Its adaptive
budget matches CUDA MSA: 7,392 to 29,568 sweeps and exactly 128 reads.
See [SA acceleration](docs/sa-acceleration.md) for the kernel and benchmark details.

Every sampler streams jobs through one shared pump, `run_stream_pump` in
`src/lib.rs`. That pump drops a cancelled generation before a worker touches
the graph. The SA kernel also polls the cancel guard once per sweep (one
`Relaxed` load, never per spin flip). A long in-flight job of 80 to 300
seconds then stops at the next sweep. The kernel emits
`StreamOutcome::Cancelled`, the same outcome as the dequeue path. The SB
kernel and its sampler live in `src/sb_core.rs` and `src/sb_sampler.rs`,
separate from the annealing path. `docs/sb-variants.md` gives the method,
the constants, and the paper behind each Simulated Bifurcation kernel, and
names every place the code departs from its paper.

Prebuilt binaries are attached to each
[Release](https://gitlab.com/quip.network/quip-solver-cpu/-/releases) for
`linux-amd64`, `linux-arm64`, and `darwin-arm64`. Asset names carry the
operating system as well as the architecture, because an architecture alone
cannot separate a Linux aarch64 binary from a Darwin arm64 one.

Every binary in the table above is published, production and experimental
alike. Each release lists the two tracks in its notes. Asset names follow the
same `<binary>-<os>-<arch>` pattern on both tracks and carry no marker, so the
file alone does not say which track it came from. Read the track column above
before you deploy one.

## Build

```sh
cargo build --release
```

A plain build produces the four production binaries. The experimental
binaries need an opt-in feature:

```sh
cargo build --release --features experimental
```

`quip-cpu-mps` chooses its bond dimension per job from a deterministic flop
budget, a 64 MB per-model memory cap, and a ceiling of 32. On graphs as wide as
`advantage2-system1` that resolves to 1, where the algorithm is mean-field
annealing rather than a tensor network. The binary degrades instead of
rejecting, because the coordinator defaults to that preset.

Set `QUIP_MPS_INIT=random` to replace the anneal with uniform random starting
configurations. Both settings share the same sampler and the same greedy polish,
which is what makes the two arms comparable.

### H3 experiment: Annealed against random initialization

`quip-cpu-mfa` reads one environment variable, `QUIP_MPS_INIT`, that switches
its starting state before the greedy polish stage. The default value, `anneal`,
lowers a transverse field through imaginary-time evolution and samples the
annealed state. The alternative value, `random`, skips the anneal and samples a
uniform random product state instead. Both values share the same sampler and the
same polish stage, so a comparison between them isolates the effect of the
seeding strategy alone.

The commands below need an `isingmark` that accepts `--backend cpu-mfa`.
`isingmark` reads that flag against a fixed built-in list, so an older build
rejects the name before it starts a run. Build the two experimental binaries
onto `PATH` first, as the Build section describes.

Hypothesis H3 asks whether the annealed seed reaches a better result than the
random seed at equal wall-clock cost. Run the two arms with the same seed and
the same single-cell parameter grid, so `isingmark` draws the same 200 problems
for both arms and the results pair by `job_id`:

```sh
# Anneal arm (default): QUIP_MPS_INIT unset.
isingmark sweep --backend cpu-mfa --topology-preset advantage2-system1 \
  --param-grid '{"num_reads":[64],"num_sweeps":[320]}' \
  --num-jobs 200 --seed 2000 --output-dir out/h3/anneal

# Random arm: QUIP_MPS_INIT=random, same grid cell, same seed.
QUIP_MPS_INIT=random isingmark sweep --backend cpu-mfa \
  --topology-preset advantage2-system1 \
  --param-grid '{"num_reads":[64],"num_sweeps":[320]}' \
  --num-jobs 200 --seed 2000 --output-dir out/h3/random
```

Repeat both commands with `--topology-preset smoke` and an
`out/h3/<arm>-smoke` output directory. The design requires that second pair as
its stop criterion.

The gap depends on the linear biases. Mean-field cannot break a symmetry on its
own. On a zero-bias instance, every coupling gate is symmetric under the global
spin flip. The annealed state stays at the symmetric point. A sample drawn there
is a fair coin per site, which makes the two arms one measurement. Biases remove
the degeneracy. H3 reports the separation that appears once they do.

The accepted-rate and paired best-energy comparison that decide H3 run through
the campaign analysis script rather than through this repository.

The Quip solver contract (`quip-proto`, `quip-protocol`, `quip-solver-core`) is
published from crates.io.

## Running

**Connect to a coordinator** (production):

```sh
quip-cpu-sa --quip-coordinator unix:///run/quip/coord.sock
```

### `num_cpus`

The coordinator `[cpu]` section may set `num_cpus`. That key travels in
`Configure.backend_toml`. The miner reads it in `apply_config`.

| Value | Effect |
|-------|--------|
| Absent | Use `available_parallelism()` as the core budget. |
| Positive integer | Use that count as the core budget. |
| Larger than the host | Clamp to the host parallelism. Log the clamp at `debug`. |
| Zero or negative | Refuse the setting. Exit 64 at handshake. |

The budget bounds sampler concurrency:

- SA: stream worker count. Each worker runs one model. Reads inside a
  model are sequential. Total threads equal `num_cpus`.
- Gibbs: stream worker count is `num_cpus / gibbs.workers`. Each in-flight
  model uses `gibbs.workers` threads (default 4). The product stays at or
  below `num_cpus`, except when `num_cpus` is smaller than `gibbs.workers`.
  Then one model still uses `gibbs.workers` threads.

- `quip-cpu-fsa` and `quip-cpu-msa`: same rule as SA. One model per core,
  models sequential inside a job, total threads equal `num_cpus`.

Other CPU binaries (SB, MPS, Flatiron) do not read `num_cpus`.

**Driver / fixed-input (run in isolation).** Use the coordinator's
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

## Python binding

`py/` holds `quip_msa`, a PyO3 binding to the multi-spin annealing kernel. It
is its own cargo workspace, and it is not published. Build it into a virtual
environment:

    cd py
    python3 -m venv .venv
    .venv/bin/pip install "maturin>=1,<2" numpy pytest
    .venv/bin/maturin develop --release
    .venv/bin/pytest tests -q

To install it into the environment of another project, activate that
environment first and run `maturin develop --release` from `py/`.

`quip_msa.Msa().sample(h, edges, j, num_sweeps=...)` anneals one Ising problem
and returns `(spins, energy_milli)` as numpy arrays. The call releases the GIL,
and the kernel runs one job on one core, so call it from several threads to
use several cores. `initial_spins` starts reads from given states, and
`start_beta` sets where a seeded anneal enters the beta ladder.

## Benchmarking

`quip-cpu-bench` runs a model through the sampler under a `tracing` subscriber
and writes per-part timing plus a flame graph. The report assigns the
sampler's time to each annealing seam, not only to the whole-model total.

```sh
quip-cpu-bench --nodes 512 --edges 2048 --iters 5 --out-dir bench-out
# or against a coordinator corpus (nonce-refs need --topology for the topology):
quip-cpu-bench --source instances.jsonl --topology topology.spec.json --out-dir bench-out
```

`--source` is the coordinator's `instances.jsonl`, one JSON object per line,
keyed on `nonce`. The bench ignores other keys. `--topology` is the
coordinator's `topology.spec.json`
(`{nodes, edges, allowed_h_milli, allowed_j_milli}`). The bench redraws each
nonce-ref entry against it with `quip_protocol::chacha8::draw_ising_milli`.
`--limit K` benches only the first `K` corpus models.

The bench times the scalar SA kernel that `quip-cpu-sa` runs. Gibbs runs on a
worker pool outside these seams, so the bench does not cover it.

Flags: `--nodes`/`--edges` (synthetic model) or
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
- `derived` — `per_spin_ns` and `accept_rate` computed as total ÷
  frequency (`sweep_loop.total_ns` ÷ spin visits), plus `sweep_loop_ns_per_read`.
- `residual_ns`/`residual_frac` — `measured_model_ns` minus the summed
  top-level parts. A large fraction signals a missing seam.
- `<model_id>.folded` — `tracing-flame` folded stacks. Render them with
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
`quip-solver-conformance` and check energies against `conformance/golden_vectors.json`.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
