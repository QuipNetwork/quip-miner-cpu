# Comparing sampler kernels

This guide describes how to evaluate a sampler kernel against the ones this
crate already ships. It states the method, the datasets, and the results of the
first run, which compared `cpu-sb` against `cpu-sa` and `cpu-gibbs`.

Follow the same method for a new kernel. A result produced a different way does
not compare to the numbers below.

## What to measure

Measure two quantities, always together:

1. **Cost.** Mean wall-clock time for one sample, where a sample is one call to
   the kernel with a fixed read count and sweep count.
2. **Quality.** The lowest energy the kernel reaches, scored with
   `quip_protocol::scoring::energy_milli`.

Neither quantity means anything alone. Less work per nominal sweep buys speed
and costs solution quality, and more work per nominal sweep does the reverse.
Report both numbers, or report nothing.

Nominal work does not transfer between kernels. One simulated annealing sweep
applies a Metropolis test to every spin. One Simulated Bifurcation step
accumulates a force and updates two vectors. Equal `num_sweeps` is not equal
work, and no correction makes it so.

## Datasets

### Pivot topology

The pivot is the D-Wave Advantage2-System1 graph, taken from the isingmark
repository:

    isingmark/fixtures/advantage2-system1.spec.json

It holds 4577 nodes and 41515 edges, which is a mean degree of 18.14. The node
identifiers are sparse, because the fixture records hardware yield gaps. The
highest identifier is 4799. Relabel the identifiers to a dense range of 0 to
`N-1` before you build a graph, and keep the ascending identifier order so the
relabelling is reproducible.

The fixture also carries the value sets the protocol draws from:
`allowed_h_milli` is `[-1000, 0, 1000]` and `allowed_j_milli` is
`[-1000, 1000]`.

### Size ladder

Expand linearly in both directions from the pivot. The first run used
quarter-pivot steps from 0.25 to 2.00, which gives 1144, 2289, 3433, 4577, 5721,
6866, 8010, and 9154 nodes.

**Below the pivot, grow the subgraph breadth-first from node 0.** Do not take
the induced subgraph on the lowest node identifiers. This graph connects low
identifiers to high ones, so an identifier cut discards most edges. The first
try made that mistake and produced a mean degree of 3.5 instead of 18. The
result measured a change in density, not a change in size. Breadth-first growth
holds the degree between 16.3 and 18.1 across the ladder.

**Above the pivot, tile whole copies of the pivot and add random cross-tile
edges** until the edge count reaches the pivot density of 9.07 edges per node.
Tiles alone leave the instance separable into independent subproblems, which no
kernel has to solve as one problem.

### Instances

Set every bias to zero. Ternary biases make the instance too easy. A per-spin
greedy pass over the biases alone already lands close to the best answer, so the
measurement reports that greedy pass rather than the kernel.

Zero biases also exercise the pure-coupling path. For Simulated Bifurcation that
path carries no ancilla particle and no ancilla force loop.

Draw each coupling uniformly from `allowed_j_milli`, which is `{-1, +1}` for
this fixture.

Generate 30 instances for each size. **Give every kernel the identical 30
instances.** Skip this step and the measurement cannot resolve the difference
between two kernels at all. See the section on reading the numbers.

### Cores and adjusted numbers

Record how many cores the kernel occupies for one sample. Report the measured
wall-clock time per sample, and beside it the throughput one core delivers:

    models/s/core = 1 / (seconds per sample * cores)

Rank on that second number. A kernel that spreads one sample over 4 cores and
finishes in half the time of a single-core kernel looks twice as fast per
sample and returns half as much per core. Wall-clock alone hides the trade.

Confirm the core count by measurement, not by reading the code. Run the kernel
under `/usr/bin/time -p` and divide user time by real time. A single-core kernel
returns a ratio near 1.

`cpu-sa` and `cpu-sb` occupy one core per sample. `cpu-gibbs` occupies
`DEFAULT_GIBBS_WORKERS`, which is 4: it hands each worker a whole read, so it
spends 4 core-seconds for every wall-clock second.
`CpuSampler::stream_width` divides the model count by the worker count to keep
the host from oversubscribing itself.

An earlier version of this crate ran Gibbs as a sequential single-site scan
despite documenting it as chromatic. That kernel is gone. Colouring is the
algorithm, not an optimisation over a scan, so no sequential Gibbs remains.

### Parameters

The first run fixed `num_reads` at 16 and `num_sweeps` at 1000 for every kernel
at every size. Both numbers are a starting point, not a tuned setting. The
adapt envelope for `cpu-sb` allows 256 to 8192 sweeps, and those bounds are
provisional.

## Results

Measured on darwin-arm64 with 12 performance and 4 efficiency cores, under
`caffeinate`. The figure is
[`comparisons/kernel_comparison.svg`](comparisons/kernel_comparison.svg). The
per-sample data is [`comparisons/results.json`](comparisons/results.json).

The `load` column is the 1-minute load average across that row. The host was
shared with other work, so treat any row above about 20 as an upper bound on
speed rather than a measurement. Energy does not depend on load and holds
everywhere. The two rows at 9154 nodes ran at load 119 and 129.

`models/s/core` is the throughput one core delivers: one divided by the
wall-clock seconds per sample, divided again by the cores the sample occupies.
Rank on that column. Wall-clock alone rewards a kernel for spending more cores,
and a miner that fills a machine with models cares what each core returns.

| kernel | nodes | edges | cores | reads | sweeps | best energy over 30 samples | mean time per sample | models/s/core | load |
|--------|------:|------:|------:|------:|-------:|----------------------------:|---------------------:|--------------:|-----:|
| cpu-sa | 1144 | 9297 | 1 | 16 | 1000 | -3405000 | 299 ms | 3.348 | 12 |
| cpu-gibbs | 1144 | 9297 | 4 | 16 | 1000 | -3385000 | 250 ms | 0.998 | 13 |
| cpu-sb | 1144 | 9297 | 1 | 16 | 1000 | -3399000 | 284 ms | 3.520 | 14 |
| cpu-sa | 2289 | 19578 | 1 | 16 | 1000 | -6968000 | 1102 ms | 0.908 | 18 |
| cpu-gibbs | 2289 | 19578 | 4 | 16 | 1000 | -6942000 | 1117 ms | 0.224 | 25 |
| cpu-sb | 2289 | 19578 | 1 | 16 | 1000 | -6972000 | 546 ms | 1.831 | 27 |
| cpu-sa | 3433 | 30556 | 1 | 16 | 1000 | -10676000 | 1653 ms | 0.605 | 29 |
| cpu-gibbs | 3433 | 30556 | 4 | 16 | 1000 | -10640000 | 1743 ms | 0.143 | 30 |
| cpu-sb | 3433 | 30556 | 1 | 16 | 1000 | -10706000 | 2375 ms | 0.421 | 26 |
| cpu-sa | 4577 | 41515 | 1 | 16 | 1000 | -14397000 | 1036 ms | 0.965 | 22 |
| cpu-gibbs | 4577 | 41515 | 4 | 16 | 1000 | -14341000 | 886 ms | 0.282 | 19 |
| cpu-sb | 4577 | 41515 | 1 | 16 | 1000 | -14399000 | 1266 ms | 0.790 | 17 |
| cpu-sa | 5721 | 51891 | 1 | 16 | 1000 | -18031000 | 1407 ms | 0.711 | 14 |
| cpu-gibbs | 5721 | 51891 | 4 | 16 | 1000 | -17947000 | 1033 ms | 0.242 | 13 |
| cpu-sb | 5721 | 51891 | 1 | 16 | 1000 | -18039000 | 1607 ms | 0.622 | 11 |
| cpu-sa | 6866 | 62277 | 1 | 16 | 1000 | -21609000 | 1565 ms | 0.639 | 12 |
| cpu-gibbs | 6866 | 62277 | 4 | 16 | 1000 | -21525000 | 1033 ms | 0.242 | 12 |
| cpu-sb | 6866 | 62277 | 1 | 16 | 1000 | -21601000 | 1873 ms | 0.534 | 13 |
| cpu-sa | 8010 | 72654 | 1 | 16 | 1000 | -24996000 | 1818 ms | 0.550 | 12 |
| cpu-gibbs | 8010 | 72654 | 4 | 16 | 1000 | -24910000 | 1253 ms | 0.199 | 11 |
| cpu-sb | 8010 | 72654 | 1 | 16 | 1000 | -25060000 | 2244 ms | 0.446 | 13 |
| cpu-sa | 9154 | 83030 | 1 | 16 | 1000 | -28694000 | 2513 ms | 0.398 | 46 |
| cpu-gibbs | 9154 | 83030 | 4 | 16 | 1000 | -28638000 | 4761 ms | 0.053 | 129 |
| cpu-sb | 9154 | 83030 | 1 | 16 | 1000 | -28756000 | 2880 ms | 0.347 | 119 |

### Paired quality difference from `cpu-sa`

A negative value means the kernel reached lower energy on the same instances,
which is better.

| nodes | cpu-sb minus cpu-sa | t | cpu-gibbs minus cpu-sa | t |
|------:|--------------------:|----:|-----------------------:|----:|
| 1144 | +0.073% | +1.9 | +0.338% | +6.0 |
| 2289 | -0.061% | -1.7 | +0.296% | +8.3 |
| 3433 | -0.090% | -2.9 | +0.288% | +9.2 |
| 4577 | -0.051% | -1.8 | +0.346% | +12.2 |
| 5721 | -0.172% | -5.6 | +0.388% | +10.8 |
| 6866 | -0.080% | -3.0 | +0.382% | +11.9 |
| 8010 | -0.173% | -5.9 | +0.326% | +8.4 |
| 9154 | -0.077% | -4.3 | +0.284% | +13.5 |

`cpu-sb` reaches lower energy than `cpu-sa` at every size above the smallest
and loses at 1144 nodes. The margin runs from 0.05 to 0.17 percent.
`cpu-gibbs` is worse at every size, by 0.28 to 0.39 percent, and every one of
those differences is many standard errors from zero.

### What the run showed

Ranked on `models/s/core`, `cpu-sa` leads at every size except the smallest,
`cpu-sb` returns 0.8 to 0.9 times as much on the clean rows, and `cpu-gibbs`
returns 0.3 to 0.4 times as much, because it occupies 4 cores to finish one
sample.

Ranked on wall-clock alone the order changes. `cpu-gibbs` finishes a sample
fastest at 5721 nodes and above, and buys that latency with 4 cores. That
inversion is the whole reason the throughput column exists.

Ranked on solution quality, `cpu-sb` is best, `cpu-sa` is close behind, and
`cpu-gibbs` is last at every size.

In short, `cpu-sb` buys about 0.1 percent lower energy for 10 to 40 percent
less throughput per core than `cpu-sa`, while `cpu-gibbs` buys nothing at a
third of the throughput per core. Whether the `cpu-sb` trade is worth taking depends on the reward
curve, and an equal-core-time comparison is still open.

### Worker scaling for chromatic Gibbs

Pivot topology, 16 reads by 1000 sweeps, 7 trials, best case:

| workers | Reads (default) | speedup | Colors | speedup |
|--------:|----------------:|--------:|-------:|--------:|
| 1 | 1.948 s | 1.00 | 1.948 s | 1.00 |
| 2 | 0.986 s | 1.97 | 1.120 s | 1.74 |
| 4 | 0.498 s | 3.91 | 0.941 s | 2.07 |
| 6 | 0.483 s | 4.03 | 0.784 s | 2.49 |
| 8 | 0.401 s | 4.86 | 0.814 s | 2.39 |
| 12 | 0.400 s | 4.87 | 0.826 s | 2.36 |
| 16 | 0.324 s | 6.02 | 1.785 s | 1.09 |

`Reads` rises monotonically to 6.02 and its slowest trial sat within 3 percent
of its fastest. `Colors` peaks at 2.65 near 5 workers, falls back, and collapses
at 16, with trials spread by up to a factor of 2.7.

The default of 4 workers reaches 3.91, which is 98 percent efficiency. Returns
fall past that point because only 12 of the 16 cores are performance cores.

Set the read count at or above the worker count before measuring `Reads`. An
earlier sweep used 4 reads, which capped it at 4 workers and understated it.

### Why colour splitting loses on a CPU

A colour class hands each worker one or two microseconds of work, and every
class ends at a barrier. Reaching even 2.65 took a persistent worker pool in
place of spawning per class, then a sense-reversing spin barrier in place of a
mutex and condvar, which had run at 0.36 times the speed of one worker.

Splitting a class is the right shape on a GPU or an FPGA, where the lane count
far exceeds the class size. On 16 CPU threads the barrier costs more than the
work it separates. `--gibbs-split-colors` selects it for comparison.

### Worker scaling for Simulated Bifurcation

Reads are independent trajectories, so splitting 16 reads across threads needs
no synchronisation. Pivot topology, median of 5:

| threads | median wall | speedup |
|--------:|------------:|--------:|
| 1 | 1.73 s | 1.00 |
| 2 | 0.90 s | 1.91 |
| 4 | 0.60 s | 2.89 |
| 8 | 0.41 s | 4.19 |
| 16 | 0.26 s | 6.54 |

Simulated Bifurcation has no ideal worker count. It scales to the core count
with falling efficiency and never collapses, because it has no barrier to
collapse at. Goto and colleagues make this the central property of the method:
every particle updates from the previous positions, so the update carries no
ordering constraint. Let the operator choose the count.

### Particle-level parallelism for Simulated Bifurcation

`sample_sb_with_workers` splits the particles inside one read. Two barriers
guard each step: one publishes the coupling vector, the other stops a fast
worker starting the next step while a slow one still reads the current one.
The output is bit-identical at every worker count, which
`parallel_matches_sequential_bit_for_bit` pins across all four variants.

One read by 1000 steps, median of 5, so the only parallelism available is
across particles:

| workers | 4577 nodes | speedup | 18308 nodes | speedup |
|--------:|-----------:|--------:|------------:|--------:|
| 1 | 140 ms | 1.00 | 429 ms | 1.00 |
| 2 | 72 ms | 1.95 | 338 ms | 1.27 |
| 4 | 56 ms | 2.48 | 277 ms | 1.55 |
| 8 | 83 ms | 1.69 | 173 ms | 2.48 |
| 16 | 660 ms | 0.21 | 783 ms | 0.55 |

Particle-level parallelism peaks near 2.5. The best worker count grows with the
problem, from 4 workers at 4577 nodes to 8 at 18308, because a larger problem
puts more work between two barriers. Oversubscription collapses this kernel as
it collapses colour splitting.

Read-level parallelism reaches 6.54 and never regresses, so prefer it whenever
the read count reaches the core count. Reach for particle-level parallelism
when the read count is the smaller number, which is where it cuts the latency
of a single read.

### Colouring the pivot topology

| property | value |
|----------|-------|
| max degree | 20 |
| bipartite | no |
| largest clique found | 4 |
| greedy, natural order | 8 classes |
| greedy, Welsh-Powell | 8 classes |
| local search at 4, 5, 6 | no colouring found |
| local search at 7, 8 | colouring found |

The chromatic number lies between 4 and 7. Greedy returns 8. Class sizes are
skewed, at 857, 850, 817, 744, 686, 477, 134 and 12, so the smallest class
caps how many workers it can occupy under `Colors`.

The class count is a property of the graph, not a setting. `GibbsConfig`
carries `max_colors` so a deployment can refuse a topology that colours worse
than it expects. It cannot request a smaller colouring.

## Reading the numbers

### Pair the comparison, or the effect disappears

The per-instance spread of the best energy a single kernel reaches is between
0.22% and 0.74%. The difference between `cpu-sb` and `cpu-sa` is between 0.05%
and 0.17%. The noise is four to ten times the effect.

Run every kernel on the same instances and take the difference instance by
instance. Instance-to-instance variance then cancels, and 30 samples resolve the
difference. Run each kernel on its own instances, and the same difference stays
buried. Two kernels that differ then look identical.

### Report the standard error

State the standard error of the paired mean, or plot it. The figure draws
plus or minus one standard error over the 30 paired instances. A difference
smaller than its own standard error is not a finding.

### Do not read the "best over 30 samples" column as the outcome

That column reports one extreme value out of 30 draws, so it moves with the
tail of the distribution. It answers the question a miner asks, which is how low
the kernel ever got. It does not support a claim that one kernel beats another.
Use the paired table for that.

## Known limits of the first run

- Single-threaded. This measures kernel cost, not miner throughput. Production
  throughput runs one model per core through the shared streaming pump, which is
  the same code for every sampler in this crate.
- Measured on one machine and one platform, so the times do not transfer.
- One parameter setting. `num_reads` at 16 and `num_sweeps` at 1000 are not
  tuned for any kernel.
- Equal nominal work, not equal time. An equal wall-clock comparison at these
  sizes is still open.
- Random cross-tile edges above the pivot. Those instances are no longer a
  hardware graph. Treat the sizes above 4577 nodes as a scaling study.

## Adding a kernel to the comparison

1. Add the kernel to the harness enumeration in
   [`comparisons/harness.rs`](comparisons/harness.rs), which needs a name and a
   call that returns the lowest energy over the reads.
2. Measure the core count. Run one sample under `/usr/bin/time -p` and divide
   user time by real time. Record the result, and divide the per-sample
   throughput by it.
3. Run the full ladder. Keep `num_reads`, `num_sweeps`, and the instance seeds
   unchanged, or the results do not compare to the preceding table.
4. Compute the paired difference against `cpu-sa` and its standard error.
5. Redraw the figure with [`comparisons/mkchart.py`](comparisons/mkchart.py).
6. Record the cost and the quality together, and state the limits that apply.

The harness reads the pivot fixture by absolute path and writes `results.json`
next to itself. It builds as a standalone binary that depends on this crate by
path. This directory keeps it as a record of the method rather than as a build
target, because it must never enter the release binaries.

## Open work

- An equal wall-clock comparison at every size on the ladder.
- A sweep of `num_sweeps` for `cpu-sb` across its adapt envelope of 256 to 8192,
  to find its operating point before any further comparison.
- A multi-threaded run through the streaming pump, to confirm that the
  single-threaded cost ratio holds at production concurrency.
