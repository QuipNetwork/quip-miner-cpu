# CPU scaling of the SA stream pump

This page answers GitHub issue #1 (Linear QUI-1407): throughput did not grow with the CPU count in a Docker container. The measurements show that the CPU SA sampler scales with the CPU quota. In a container with `--cpus=10`, the sampler completes 1.93 jobs/s for the reported job shape on the test host.

## Method

- Benchmark: `examples/stream_scaling.rs`. It drives `CpuSampler::new(Algorithm::Sa)` through `sample_stream`, the same path that the miner uses. It sets `num_cpus` through `backend_toml` and submits 4 jobs for each requested worker.
- Timing: the benchmark stamps each result on arrival, before it checks the result. Wall time ends at the last arrival. A job's latency includes the time that the job waits in the queue.
- Job shape: the Advantage2 topology (4,577 nodes, 41,515 edges) with random ±1 couplings, 105 reads, and 846 sweeps. These values match the progress line in the issue.
- Host: an AMD Ryzen 9 5950X with 16 cores and 32 hardware threads. The 1-minute load average was 1.2 before the run. During the 32-worker runs the benchmark itself raised it to 26.
- Containers: `rust:1.98.1-bookworm` with `docker run --cpus=4` and `--cpus=10`, running the same benchmark binary.
- Data and the run script: `quip-data/cpu-scaling/2026-10-01/`.

The topology comes from `isingmark/fixtures/advantage2-system1.spec.json`. The miner's coordinator fixtures carry the same file at `quip-miner/crates/quip-coordinator/fixtures/drive/advantage2-system1.spec.json`. Pass its path to the benchmark:

```sh
cargo run --release --example stream_scaling -- --topology <path>/advantage2-system1.spec.json
```

## Results

One job alone on one worker takes 4.4 seconds (3 runs: 4.41, 4.47, and 4.49 seconds).

The "workers" columns show the worker count that the sampler started. In a container the sampler clamps the requested count to the CPU quota. Service time is workers divided by jobs/s. It is the average time that one worker spends on one job.

| Requested | Host workers | Host jobs/s | Host service s | `--cpus=10` workers | `--cpus=10` jobs/s | `--cpus=4` workers | `--cpus=4` jobs/s |
| -- | -- | -- | -- | -- | -- | -- | -- |
| 1 | 1 | 0.23 | 4.4 | 1 | 0.22 | 1 | 0.22 |
| 2 | 2 | 0.45 | 4.5 | 2 | 0.44 | 2 | 0.42 |
| 4 | 4 | 0.88 | 4.6 | 4 | 0.86 | 4 | 0.83 |
| 8 | 8 | 1.67 | 4.8 | 8 | 1.61 | 4 | 0.85 |
| 10 | 10 | 1.96 | 5.1 | 10 | 1.93 | 4 | 0.82 |
| 14 | 14 | 2.63 | 5.3 | 10 | 1.86 | 4 | 0.85 |
| 16 | 16 | 2.53 | 6.3 | 10 | 1.79 | 4 | 0.85 |
| 28 | 28 | 3.34 | 8.4 | 10 | 1.80 | 4 | 0.86 |
| 32 | 32 | 3.44 | 9.3 | 10 | 1.90 | 4 | 0.83 |

- Host throughput grows almost linearly up to 14 workers. At 14 workers each job takes 20% longer than a job alone, for a cause that this run did not measure.
- From 16 to 32 workers, the second hardware thread on each core adds 36% more throughput. At 32 workers each job takes about twice as long as a job alone.
- In a container, `std::thread::available_parallelism()` returns the CPU quota: 4 for `--cpus=4` and 10 for `--cpus=10`. The worker count never exceeds the quota.
- The per-job latency, including queue time, is in the `median_job_s` and `p90_job_s` columns of the CSV files.

## Answers to the questions in the issue

**1. Is jobs/s the throughput of all workers together?** Yes. The miner prints completed jobs from all workers, divided by the time since the session started. The time includes startup and every wait for new jobs from the coordinator. Early in a session, or when jobs arrive late, the number is lower than the sampling rate.

**2. Is 0.9 jobs/s expected with 10 CPUs?** Not on this host. With 10 workers, the ideal rate is 10 / 4.4 s, or 2.3 jobs/s. The measured rate in a `--cpus=10` container is 1.93 jobs/s, because each job runs 16% slower when 10 run at once. A rate of 0.9 jobs/s with 10 busy workers means each job takes about 11 seconds. A slower core, or 10 hardware threads on 5 physical cores, can cause that.

**3. Can you see the worker count and per-worker throughput?** The command `quip-cpu-sa --capabilities` prints `streamWidth`. That value is the declared worker count from the host or container CPU count. A `num_cpus` setting lower than that count starts fewer workers, so `streamWidth` is an upper bound. Every finished job logs an `attempt` line with two times. The `device` time is the time that one worker spent sampling the job, so 1 divided by it is the per-worker rate. The `wall` time also includes the time that the job waited in the queue.

**4. Is there a bottleneck that prevents scaling?** This measurement did not find one in the SA sampler. The worker count follows the container quota, and throughput grows almost linearly up to the number of physical cores. Worker sizing in 0.3.2-rc3 uses the same code. To find the cause on your host, send these items:
   - The CPU model and the output of `nproc` in the container.
   - The full `docker run` CPU flags (`--cpus` or `--cpuset-cpus`).
   - Your `num_cpus` setting, if you set one.
   - A few `attempt` log lines with their `wall` and `device` times.
