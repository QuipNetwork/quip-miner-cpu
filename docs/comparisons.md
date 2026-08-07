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
wall-clock time first, then the core-adjusted time in parentheses, where the
adjusted time is the wall-clock time multiplied by the core count. The adjusted
number is the cost in core-seconds, which is what a miner pays when it runs one
model per core.

A kernel that spreads one sample over 4 cores and finishes in half the time of a
single-core kernel is twice as fast per sample and twice as expensive per core.
Only the adjusted number exposes that.

Confirm the core count by measurement, not by reading the code. Run the kernel
under `/usr/bin/time -p` and divide user time by real time. A single-core kernel
returns a ratio near 1.

`cpu-sa` and `cpu-sb` occupy one core per sample. `cpu-gibbs` occupies
`DEFAULT_GIBBS_WORKERS`, which is 4: chromatic Gibbs splits each colour class
across workers, so it spends 4 core-seconds for every wall-clock second.
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

## Results of the first run

Measured on darwin-arm64 with 12 performance and 4 efficiency cores. The figure is
[`comparisons/kernel_comparison.svg`](comparisons/kernel_comparison.svg). The
per-sample data is [`comparisons/results.json`](comparisons/results.json).

| kernel | nodes | edges | cores | reads | sweeps | best energy over 30 samples | mean time per sample |
|--------|------:|------:|------:|------:|-------:|----------------------------:|---------------------:|
| cpu-sa | 1144 | 9297 | 1 | 16 | 1000 | -3405000 | 234 ms |
| cpu-gibbs | 1144 | 9297 | 1 | 16 | 1000 | -3389000 | 319 ms |
| cpu-sb | 1144 | 9297 | 1 | 16 | 1000 | -3399000 | 243 ms |
| cpu-sa | 2289 | 19578 | 1 | 16 | 1000 | -6968000 | 512 ms |
| cpu-gibbs | 2289 | 19578 | 1 | 16 | 1000 | -6950000 | 644 ms |
| cpu-sb | 2289 | 19578 | 1 | 16 | 1000 | -6972000 | 533 ms |
| cpu-sa | 3433 | 30556 | 1 | 16 | 1000 | -10676000 | 757 ms |
| cpu-gibbs | 3433 | 30556 | 1 | 16 | 1000 | -10664000 | 941 ms |
| cpu-sb | 3433 | 30556 | 1 | 16 | 1000 | -10706000 | 824 ms |
| cpu-sa | 4577 | 41515 | 1 | 16 | 1000 | -14397000 | 1012 ms |
| cpu-gibbs | 4577 | 41515 | 1 | 16 | 1000 | -14377000 | 1330 ms |
| cpu-sb | 4577 | 41515 | 1 | 16 | 1000 | -14399000 | 1281 ms |
| cpu-sa | 5721 | 51891 | 1 | 16 | 1000 | -18031000 | 1255 ms |
| cpu-gibbs | 5721 | 51891 | 1 | 16 | 1000 | -17985000 | 1677 ms |
| cpu-sb | 5721 | 51891 | 1 | 16 | 1000 | -18039000 | 1535 ms |
| cpu-sa | 6866 | 62277 | 1 | 16 | 1000 | -21609000 | 1532 ms |
| cpu-gibbs | 6866 | 62277 | 1 | 16 | 1000 | -21493000 | 2116 ms |
| cpu-sb | 6866 | 62277 | 1 | 16 | 1000 | -21601000 | 1817 ms |
| cpu-sa | 8010 | 72654 | 1 | 16 | 1000 | -24996000 | 1777 ms |
| cpu-gibbs | 8010 | 72654 | 1 | 16 | 1000 | -24870000 | 2363 ms |
| cpu-sb | 8010 | 72654 | 1 | 16 | 1000 | -25060000 | 2268 ms |
| cpu-sa | 9154 | 83030 | 1 | 16 | 1000 | -28694000 | 1879 ms |
| cpu-gibbs | 9154 | 83030 | 1 | 16 | 1000 | -28636000 | 2596 ms |
| cpu-sb | 9154 | 83030 | 1 | 16 | 1000 | -28756000 | 2596 ms |

**These timings predate the Gibbs rewrite and are not current.** The energy
columns still hold, because energy does not depend on machine load. The
wall-clock column does not: a later rerun moved `cpu-sa`, which did not change
at all, from 1012 ms to 2191 ms at the pivot. Rerun the ladder on a quiet
machine before quoting any time from it.

A controlled measurement at the pivot, 16 reads by 1000 sweeps, 7 trials:

| kernel | cores | min | median | max | best energy |
|--------|------:|----:|-------:|----:|------------:|
| cpu-sa | 1 | 1529 ms | 1604 ms | 1652 ms | -14319000 |
| cpu-gibbs | 4 | 1738 ms | 2989 ms | 7300 ms | -14303000 |
| cpu-sb | 1 | 1894 ms | 1970 ms | 2175 ms | -14323000 |

`cpu-gibbs` spans a factor of 4.2 between its fastest and slowest trial while
the single-core kernels stay inside 8 percent. See the section on barrier cost.

### Paired quality difference from `cpu-sa`

A negative value means the kernel reached lower energy than `cpu-sa` on the same
instances, which is better.

| nodes | cpu-sb minus cpu-sa | t | instances where cpu-sb won |
|------:|--------------------:|----:|---------------------------:|
| 1144 | +0.073% | +1.86 | 8 of 30 |
| 2289 | -0.061% | -1.74 | 17 of 30 |
| 3433 | -0.090% | -2.89 | 20 of 30 |
| 4577 | -0.051% | -1.78 | 18 of 30 |
| 5721 | -0.172% | -5.57 | 25 of 30 |
| 6866 | -0.080% | -3.02 | 22 of 30 |
| 8010 | -0.173% | -5.94 | 27 of 30 |
| 9154 | -0.077% | -4.33 | 23 of 30 |

`cpu-gibbs` held between +0.22% and +0.44% at every size. It is slower than
both other kernels and never reached lower energy.

### What the first run showed

`cpu-sb` costs more time per sample than `cpu-sa` at the pivot and above: 1.26
times at 4577 nodes and 1.38 times at 9154 nodes. Below about 2300 nodes the two
cost the same.

`cpu-sb` reaches lower energy than `cpu-sa` at every size above the smallest,
and loses at the smallest. The margin is between 0.05% and 0.17%.

The trade is about 0.1% lower energy for about 30% more time at pivot
scale. Whether that trade is worth taking depends on the reward curve. At equal
wall-clock time `cpu-sa` completes about 1.3 times as many reads, and more reads
also lower the energy a kernel reaches.

### Worker scaling for chromatic Gibbs

Pivot topology, 4 reads by 1000 sweeps, median of 5:

| workers | median wall | speedup |
|--------:|------------:|--------:|
| 1 | 0.85 s | 1.00 |
| 2 | 0.42 s | 2.00 |
| 4 | 0.25 s | 3.38 |
| 8 | 0.30 s | 2.79 |
| 12 | 0.27 s | 3.12 |
| 16 | 1.93 s | 0.44 |

Four workers is the best setting, at 85 percent efficiency. Sixteen
oversubscribes a 16-thread host and collapses, so `GibbsConfig::validate`
refuses a worker count above the reported core count.

### Barrier cost limits chromatic Gibbs

Reaching even that speedup took two changes, and the result is still unstable.

A class update is one or two microseconds of work per worker, and every class
ends at a barrier. Spawning threads per class cost more than the work itself:
8 classes over 1000 sweeps is 8000 spawn-and-join cycles per read, and
throughput fell as workers rose. A persistent pool fixed that. A mutex and
condvar barrier was still too expensive for a class this small, and four
workers ran at 0.36 times the speed of one. A sense-reversing spin barrier with a
bounded spin and a yield fallback is what produces the table above.

The remaining variance is scheduling. When the operating system moves one
worker to an efficiency core, the other three wait at the barrier. One such
placement costs a great deal across 128000 barriers per sample. Report the Gibbs timings as a distribution with its spread, never as a single
number.

### Worker scaling for Simulated Bifurcation

Reads are independent trajectories, so splitting 16 reads across threads needs
no synchronisation and no kernel change. Pivot topology, median of 5:

| threads | median wall | speedup |
|--------:|------------:|--------:|
| 1 | 1.73 s | 1.00 |
| 2 | 0.90 s | 1.91 |
| 4 | 0.60 s | 2.89 |
| 8 | 0.41 s | 4.19 |
| 16 | 0.26 s | 6.54 |

Simulated Bifurcation has no ideal worker count. It scales monotonically to the
core count with falling efficiency and never collapses, because it has no
barrier to collapse at. Goto and colleagues make this the central property of
the method: every particle updates from the previous positions, so the update
carries no ordering constraint. Let the operator choose the count.

This measures parallelism across reads only. The papers emphasise a second
kind, across the particles inside one read, which starts to matter once the
read count falls below the core count. No code here does that, so it stays
unmeasured.

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
caps how many workers it can occupy.

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
   user time by real time. Record the result, and report the core-adjusted time
   in parentheses when the count is above 1.
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
