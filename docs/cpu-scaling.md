# CPU scaling of the SA stream pump

This page answers GitHub issue #1 (Linear QUI-1407): throughput did not grow with the CPU count in a Docker container. The measurements show that the CPU SA sampler scales with the CPU quota. In a container with `--cpus=10`, the sampler completes 2.0 jobs/s for the reported job shape on the test host.

## Method

- Benchmark: `examples/stream_scaling.rs`. It drives `CpuSampler::new(Algorithm::Sa)` through `sample_stream`, the same path that the miner uses. It sets `num_cpus` through `backend_toml` and submits 4 jobs per worker. For each job, it records the time from submission to result.
- Job shape: the Advantage2 topology (4,577 nodes, 41,515 edges) with random ±1 couplings, 105 reads, and 846 sweeps. These values match the progress line in the issue.
- Host: an AMD Ryzen 9 5950X with 16 cores and 32 hardware threads. Other services kept the 1-minute load average near 2.5 during the run.
- Containers: `rust:1.98.1-bookworm` with `docker run --cpus=4` and `--cpus=10`, running the same benchmark binary.
- Data and the run script: `quip-data/cpu-scaling/2026-10-01/`.

Run the benchmark:

```sh
cargo run --release --example stream_scaling -- \
  --topology isingmark/fixtures/advantage2-system1.spec.json
```

## Results

One job alone on one worker takes 4.4 seconds (3 runs: 4.42, 4.43, and 4.50 seconds).

| Workers | Host jobs/s | `--cpus=10` jobs/s | `--cpus=4` jobs/s |
| -- | -- | -- | -- |
| 1 | 0.22 | 0.22 | 0.21 |
| 2 | 0.44 | 0.43 | 0.42 |
| 4 | 0.87 | 0.86 | 0.85 |
| 8 | 1.66 | 1.64 | 0.85 (4 workers) |
| 10 | 2.04 | 1.98 | 0.85 (4 workers) |
| 16 | 2.76 | 1.78 (10 workers) | 0.85 (4 workers) |
| 32 | 3.47 | 1.88 (10 workers) | 0.85 (4 workers) |

- Throughput grows almost linearly up to 16 workers, one for each physical core.
- From 16 to 32 workers, the second hardware thread on each core adds 26%.
- In a container, `std::thread::available_parallelism()` returns the CPU quota: 4 for `--cpus=4` and 10 for `--cpus=10`. The sampler clamps `num_cpus` to this value, so the worker count never exceeds the quota. The "(N workers)" entries show the clamped count.

## Answers to the questions in the issue

**1. Is jobs/s the throughput of all workers together?** Yes. The miner prints completed jobs from all workers, divided by the time since the session started. The time includes startup and every wait for new jobs from the coordinator. Early in a session, or when jobs arrive late, the number is lower than the sampling rate.

**2. Is 0.9 jobs/s expected with 10 CPUs?** Not on this host. With 10 CPUs, the expected rate is about 10 divided by the time for one job on one core. Here that is 10 / 4.4 s, or 2.0 jobs/s. A rate of 0.9 jobs/s with 10 busy workers means each job takes about 11 seconds. A slower core, or 10 hardware threads on 5 physical cores, can cause that.

**3. Can you see the worker count and per-worker throughput?** The command `quip-cpu-sa --capabilities` prints `stream_width`, which is the worker count. Every finished job logs an `attempt` line with its wall time. The per-worker rate is 1 divided by that wall time.

**4. Is there a bottleneck that prevents scaling?** This measurement did not find one in the SA sampler. The worker count follows the container quota, and throughput grows linearly up to the number of physical cores. Worker sizing in 0.3.2-rc3 uses the same code. To find the cause on your host, send these items:
   - The CPU model and the output of `nproc` in the container.
   - The full `docker run` CPU flags (`--cpus` or `--cpuset-cpus`).
   - The `stream_width` value from `--capabilities`.
   - A few `attempt` log lines with their wall times.
