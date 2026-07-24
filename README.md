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

## Tests

```sh
cargo test --release
```

Conformance/golden and handshake tests drive the binary in isolation via
`quip-mock-coordinator` and check energies against `conformance/golden_vectors.json`.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
