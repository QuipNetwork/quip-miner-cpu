# quip-miner-cpu

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
| `quip-cpu-bsb` | ballistic Simulated Bifurcation | experimental |
| `quip-cpu-hdsb` | heated discrete Simulated Bifurcation | experimental |
| `quip-cpu-hbsb` | heated ballistic Simulated Bifurcation | experimental |
| `quip-cpu-mps` | tensor network: imaginary-time TEBD with exact sampling | experimental |
| `quip-cpu-mfa` | mean-field annealing (the same kernel at bond dimension 1) | experimental |
| `quip-cpu-flatiron` | belief-propagation tensor network on the problem graph | experimental |

Every sampler streams jobs through one shared pump, `run_stream_pump` in
`src/lib.rs`. That pump holds the only copy of the cancellation check before
dispatch and of the panic re-raise on worker join. The SB kernel and its
sampler live in `src/sb_core.rs` and `src/sb_sampler.rs`, separate from the
annealing path.

Prebuilt binaries are attached to each
[Release](https://gitlab.com/quip.network/quip-miner-cpu/-/releases) for
`linux-amd64`, `linux-arm64`, and `darwin-arm64`. Asset names carry the
operating system as well as the architecture, because an architecture alone
cannot separate a Linux aarch64 binary from a Darwin arm64 one.

## Build

```sh
cargo build --release        # needs protoc on PATH (protobuf-compiler)
```

The experimental binaries build behind an opt-in feature and never appear in
a Release:

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

## Tests

```sh
cargo test --release
```

Conformance/golden and handshake tests drive the binary in isolation via
`quip-mock-coordinator` and check energies against `conformance/golden_vectors.json`.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
