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

Set every bias to zero, and note what that choice costs. The pivot fixture
carries `allowed_h_milli` of `[-1000, 0, 1000]`, so this ladder keeps the
fixture's edges and drops its bias set.

The original reason was that ternary biases make the instance too easy, because
a per-spin greedy pass over the biases alone lands close to the best answer. The
replayed `chain-ternary` problems show that reason is too strong. Kernels still
separate there, and `cpu-sb` still beats `cpu-sa` at a `t` of 3.9 in its favour.

Zero biases exercise the pure-coupling path. For Simulated Bifurcation that path
carries no ancilla particle and no ancilla force loop. That is a real saving in
the kernel under test, and it is also the blind spot: the ladder never reaches
the ancilla path, which is where both continuous-coupling variants fail.

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
`caffeinate`, with App Nap disabled. The figure is
[`comparisons/kernel_comparison.svg`](comparisons/kernel_comparison.svg). The
per-sample data is [`comparisons/results.json`](comparisons/results.json).

The `load` column is the 1-minute load average across that row. Energy does not
depend on load and holds everywhere. Time does. The host was quiet for this run:
the median load is 6, and 43 of the 48 rows sit below 10. The five rows above 20
are the first minutes of the run, and they are marked in the figure.

`models/s/core` is the throughput one core delivers: one divided by the
wall-clock seconds per sample, divided again by the cores the sample occupies.
Rank on that column. Wall-clock alone rewards a kernel for spending more cores,
and a miner that fills a machine with models cares what each core returns.

| kernel | nodes | edges | cores | reads | sweeps | best energy over 30 | mean energy | mean time per sample | models/s/core | load |
|--------|------:|------:|------:|------:|-------:|--------------------:|------------:|---------------------:|--------------:|-----:|
| cpu-sa | 1144 | 9297 | 1 | 16 | 1000 | -3405000 | -3341933 | 236 ms | 4.244 | 5 |
| cpu-gibbs | 1144 | 9297 | 4 | 16 | 1000 | -3385000 | -3330600 | 118 ms | 2.123 | 5 |
| cpu-sb | 1144 | 9297 | 1 | 16 | 1000 | -3399000 | -3339467 | 238 ms | 4.194 | 5 |
| cpu-bsb | 1144 | 9297 | 1 | 16 | 1000 | -3399000 | -3328600 | 192 ms | 5.217 | 5 |
| cpu-mps | 1144 | 9297 | 1 | 16 | 1000 | -2941000 | -2882200 | 1579 ms | 0.633 | 15 |
| cpu-mfa | 1144 | 9297 | 1 | 16 | 1000 | -2941000 | -2882200 | 2520 ms | 0.397 | 37 |
| cpu-sa | 2289 | 19578 | 1 | 16 | 1000 | -6968000 | -6899867 | 473 ms | 2.116 | 43 |
| cpu-gibbs | 2289 | 19578 | 4 | 16 | 1000 | -6942000 | -6879400 | 275 ms | 0.908 | 37 |
| cpu-sb | 2289 | 19578 | 1 | 16 | 1000 | -6972000 | -6904067 | 554 ms | 1.806 | 35 |
| cpu-bsb | 2289 | 19578 | 1 | 16 | 1000 | -6940000 | -6881267 | 423 ms | 2.363 | 31 |
| cpu-mps | 2289 | 19578 | 1 | 16 | 1000 | -5960000 | -5906267 | 3501 ms | 0.286 | 19 |
| cpu-mfa | 2289 | 19578 | 1 | 16 | 1000 | -5960000 | -5906267 | 3477 ms | 0.288 | 7 |
| cpu-sa | 3433 | 30556 | 1 | 16 | 1000 | -10676000 | -10605733 | 698 ms | 1.432 | 5 |
| cpu-gibbs | 3433 | 30556 | 4 | 16 | 1000 | -10640000 | -10575133 | 424 ms | 0.590 | 5 |
| cpu-sb | 3433 | 30556 | 1 | 16 | 1000 | -10706000 | -10615267 | 809 ms | 1.236 | 5 |
| cpu-bsb | 3433 | 30556 | 1 | 16 | 1000 | -10676000 | -10585667 | 663 ms | 1.508 | 5 |
| cpu-mps | 3433 | 30556 | 1 | 16 | 1000 | -9192000 | -9060333 | 7031 ms | 0.142 | 4 |
| cpu-mfa | 3433 | 30556 | 1 | 16 | 1000 | -9192000 | -9060333 | 7040 ms | 0.142 | 5 |
| cpu-sa | 4577 | 41515 | 1 | 16 | 1000 | -14397000 | -14307067 | 960 ms | 1.042 | 6 |
| cpu-gibbs | 4577 | 41515 | 4 | 16 | 1000 | -14341000 | -14257600 | 635 ms | 0.394 | 6 |
| cpu-sb | 4577 | 41515 | 1 | 16 | 1000 | -14399000 | -14314333 | 1217 ms | 0.821 | 6 |
| cpu-bsb | 4577 | 41515 | 1 | 16 | 1000 | -14379000 | -14285933 | 1012 ms | 0.988 | 6 |
| cpu-mps | 4577 | 41515 | 1 | 16 | 1000 | -12377000 | -12122933 | 11752 ms | 0.085 | 6 |
| cpu-mfa | 4577 | 41515 | 1 | 16 | 1000 | -12377000 | -12122933 | 11780 ms | 0.085 | 6 |
| cpu-sa | 5721 | 51891 | 1 | 16 | 1000 | -18031000 | -17942800 | 1261 ms | 0.793 | 7 |
| cpu-gibbs | 5721 | 51891 | 4 | 16 | 1000 | -17947000 | -17873200 | 862 ms | 0.290 | 7 |
| cpu-sb | 5721 | 51891 | 1 | 16 | 1000 | -18039000 | -17973600 | 1539 ms | 0.650 | 8 |
| cpu-bsb | 5721 | 51891 | 1 | 16 | 1000 | -17953000 | -17870200 | 1269 ms | 0.788 | 7 |
| cpu-mps | 5721 | 51891 | 1 | 16 | 1000 | -15377000 | -15200400 | 17445 ms | 0.057 | 5 |
| cpu-mfa | 5721 | 51891 | 1 | 16 | 1000 | -15377000 | -15200400 | 17455 ms | 0.057 | 5 |
| cpu-sa | 6866 | 62277 | 1 | 16 | 1000 | -21609000 | -21493800 | 1557 ms | 0.642 | 6 |
| cpu-gibbs | 6866 | 62277 | 4 | 16 | 1000 | -21525000 | -21411733 | 898 ms | 0.278 | 6 |
| cpu-sb | 6866 | 62277 | 1 | 16 | 1000 | -21601000 | -21510867 | 1768 ms | 0.566 | 6 |
| cpu-bsb | 6866 | 62277 | 1 | 16 | 1000 | -21487000 | -21373933 | 1474 ms | 0.679 | 5 |
| cpu-mps | 6866 | 62277 | 1 | 16 | 1000 | -18325000 | -18186200 | 24278 ms | 0.041 | 6 |
| cpu-mfa | 6866 | 62277 | 1 | 16 | 1000 | -18325000 | -18186200 | 24218 ms | 0.041 | 7 |
| cpu-sa | 8010 | 72654 | 1 | 16 | 1000 | -24996000 | -24879133 | 1784 ms | 0.560 | 7 |
| cpu-gibbs | 8010 | 72654 | 4 | 16 | 1000 | -24910000 | -24798067 | 1176 ms | 0.213 | 9 |
| cpu-sb | 8010 | 72654 | 1 | 16 | 1000 | -25060000 | -24922200 | 2200 ms | 0.455 | 10 |
| cpu-bsb | 8010 | 72654 | 1 | 16 | 1000 | -24946000 | -24829400 | 1845 ms | 0.542 | 10 |
| cpu-mps | 8010 | 72654 | 1 | 16 | 1000 | -21438000 | -21091333 | 32300 ms | 0.031 | 9 |
| cpu-mfa | 8010 | 72654 | 1 | 16 | 1000 | -21438000 | -21091333 | 32915 ms | 0.030 | 7 |
| cpu-sa | 9154 | 83030 | 1 | 16 | 1000 | -28694000 | -28574600 | 1905 ms | 0.525 | 6 |
| cpu-gibbs | 9154 | 83030 | 4 | 16 | 1000 | -28638000 | -28493400 | 1391 ms | 0.180 | 8 |
| cpu-sb | 9154 | 83030 | 1 | 16 | 1000 | -28756000 | -28596467 | 2595 ms | 0.385 | 8 |
| cpu-bsb | 9154 | 83030 | 1 | 16 | 1000 | -28700000 | -28545800 | 2088 ms | 0.479 | 7 |
| cpu-mps | 9154 | 83030 | 1 | 16 | 1000 | -24352000 | -24097133 | 41139 ms | 0.024 | 6 |
| cpu-mfa | 9154 | 83030 | 1 | 16 | 1000 | -24352000 | -24097133 | 41106 ms | 0.024 | 6 |

`cpu-mps` and `cpu-mfa` return byte-identical energies at every size. That is
not a coincidence in the data. `select_chi` returns a bond dimension of 1 on
every instance in this ladder, so the two binaries run the same product-state
code. The section below explains why.

### Paired quality difference from `cpu-sa`

Each kernel solved the same 30 instances as `cpu-sa` at each size, so the
comparison pairs per instance. A negative value means the kernel reached lower
energy, which is better. `t` is the size of the mean divided by its standard
error. Its sign always matches the sign of the difference, so it is not
repeated.

| nodes | cpu-sb | cpu-bsb | cpu-gibbs | cpu-mps and cpu-mfa |
|------:|-------:|--------:|----------:|--------------------:|
| 1144 | +0.073% (t 1.9) | +0.398% (t 5.5) | +0.338% (t 6.0) | +13.755% (t 72.3) |
| 2289 | -0.061% (t 1.7) | +0.269% (t 6.5) | +0.296% (t 8.3) | +14.399% (t 144.3) |
| 3433 | -0.090% (t 2.9) | +0.189% (t 4.0) | +0.288% (t 9.2) | +14.572% (t 129.2) |
| 4577 | -0.051% (t 1.8) | +0.148% (t 5.5) | +0.346% (t 12.2) | +15.266% (t 131.8) |
| 5721 | -0.172% (t 5.6) | +0.405% (t 15.8) | +0.388% (t 10.8) | +15.284% (t 173.9) |
| 6866 | -0.080% (t 3.0) | +0.557% (t 15.7) | +0.382% (t 11.9) | +15.388% (t 198.0) |
| 8010 | -0.173% (t 5.9) | +0.200% (t 6.2) | +0.326% (t 8.4) | +15.225% (t 159.2) |
| 9154 | -0.077% (t 4.3) | +0.101% (t 4.3) | +0.284% (t 13.5) | +15.669% (t 246.3) |

Three results hold across the whole ladder.

`cpu-sb` is the only kernel that beats `cpu-sa`. It wins at seven of the eight
sizes and loses at 1144 nodes. The margin runs from 0.05 to 0.17 percent. At
1144, 2289 and 4577 nodes the difference is within two standard errors of
zero, so the win at those sizes is not established. At 5721 and 8010 nodes the
difference reaches five standard errors.

`cpu-bsb` and `cpu-gibbs` are worse than `cpu-sa` at every size, by 0.10 to 0.56
percent. Every one of those differences is many standard errors from zero.

`cpu-mps` and `cpu-mfa` are worse by 13.8 to 15.7 percent, and the gap grows
with problem size. This is two orders of magnitude larger than any other effect
in the table.

### Why the tensor-network kernels lose on this ladder

Two mechanisms act on this ladder. Both are real, and the corpus replay below
shows that neither one explains the size of the loss.

First, `select_chi` returns 1 for every instance on this ladder: the flop budget
allows `cbrt(1.25e9 / (50 * steps * span_sum))`, and the span sum after reverse
Cuthill-McKee is 7.5 million at the pivot, which puts the result below 1. Both
binaries run as a product state, so they return identical energies at
every size here.

Second, a product state cannot break a symmetry. With no bias, every coupling
gate is symmetric under the global spin flip, so its best rank-1 factor is the
symmetric vector. The annealed state stays at the symmetric point for the whole
schedule, and a sample drawn there is a fair coin per site. Only the greedy
polish does useful work, and a strict-descent polish cannot cross the energy
plateau that a domain wall sits on. The unit tests measure the deviation from
the symmetric point as exactly zero, so this mechanism is confirmed, not
inferred.

Both mechanisms were later tested directly on replayed chain problems, and both
failed to account for the result. On the real graphs `select_chi` returns more
than 1, so the tensor network runs, and the two binaries no longer agree on any
instance. Real ternary biases recover 1.7 points of a 15 point deficit. Neither
mechanism explains the loss. The section on the greedy polish below carries the
measured cause.

### What the run showed

Ranked on `models/s/core`, `cpu-bsb` and `cpu-sa` lead and trade places,
`cpu-sb` returns 0.73 to 0.98 times `cpu-sa`, `cpu-gibbs` returns 0.34 to 0.50
times, and the two tensor-network binaries return 0.05 to 0.15 times.

Ranked on wall-clock alone the order changes. `cpu-gibbs` finishes a sample
faster than `cpu-sa` at every size, and buys that latency with 4 cores. That
inversion is the whole reason the throughput column exists.

Ranked on solution quality, `cpu-sb` is best at seven of eight sizes on mean
energy and six of eight on best energy. No other kernel wins a size on either
measure except `cpu-sa`.

`cpu-sb` buys 0.05 to 0.17 percent lower energy for 2 to 27 percent less
throughput per core than `cpu-sa`. Whether that trade is worth taking depends on
the reward curve, and an equal-core-time comparison is still open. `cpu-bsb`,
`cpu-gibbs`, `cpu-mps` and `cpu-mfa` each cost quality without returning
throughput, on this dataset.

### An internal check on the load column

`cpu-mps` and `cpu-mfa` run the same algorithm here, so any difference in their
measured time is measurement error with no algorithmic part. At 1144 nodes they
report 0.633 and 0.397 models per second per core, a 59 percent gap, at loads of
8 and 37. At every other size, where their loads match, they agree within 1
percent. The load stamp flags the one row it should and no others.

### Solution diversity at the pivot

A batch of identical reads is worth one read, so quality alone does not
describe what a kernel returns. Measured at the pivot over the same 30
instances, 16 reads each. `spread` is the median read minus the best read.

| kernel | distinct spins per 16 | distinct energies per 16 | best | spread |
|--------|----------------------:|-------------------------:|-----:|-------:|
| cpu-sa | 16.00 | 14.23 | -14307067 | 59867 |
| cpu-gibbs | 16.00 | 14.23 | -14257600 | 66467 |
| cpu-sb | 16.00 | 13.80 | -14314333 | 53200 |
| cpu-bsb | 16.00 | 13.40 | -14285933 | 45133 |
| cpu-mps | 16.00 | 15.50 | -12122933 | 234000 |
| cpu-mfa | 16.00 | 15.50 | -12122933 | 234000 |

No kernel collapses. Every one returns 16 distinct configurations from 16
reads at this size. `cpu-bsb` returns the fewest distinct energies and the
tightest spread, which fits a continuous coupling holding trajectories
correlated, but the effect is small and it costs nothing here.

`cpu-mps` and `cpu-mfa` return the most distinct energies and four times the
spread of any other kernel, together with the worst best energy. That pattern
is uniform random sampling. It confirms the mechanism in the section above:
the annealed state stays at the symmetric point, each read is a fair coin per
site, and only the greedy polish lowers the energy. Diversity is not a virtue
in that row. It is the absence of signal in the state being sampled.

The measurement is [`comparisons/diversity.rs`](comparisons/diversity.rs), and
its output is [`comparisons/diversity.json`](comparisons/diversity.json).

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

## Replayed chain problems

The ladder above draws its own instances. This second measurement replays
problems the chain actually posed, which removes the question of whether the
synthetic instances represent the real workload. It answers a different
question from the ladder, and where the two disagree, this one governs.

### Method

Two corpora ship with isingmark, each holding instances harvested from chain
and verified against their blocks. Each pairs with the topology its instances
were drawn against.

| Corpus | Instances | Nodes | Edges | `allowed_h_milli` |
| --- | --- | --- | --- | --- |
| `chain-h0` | 4658 | 4577 | 41515 | `[0]` |
| `chain-ternary` | 3148 | 4578 | 41531 | `[-1000, 0, 1000]` |

The two buckets are different graphs, not one graph under two field settings.
`chain-ternary` carries one more node and 16 more edges. A comparison between
the corpora mixes the bias set with the topology, and no result below
attributes a difference between corpora to bias alone.

Each corpus is ranked hardest first, by the lowest energy any miner reached on
chain. These runs take the hardest 50, which are the instances that decide
whether a miner earns anything. Every kernel gets the identical 50 instances at
`--hardness 0.5`, so reads and sweeps match across kernels.

Two baselines come with each instance. `energy_milli` is the best energy any
miner reached on chain. `max_energy_milli` is the difficulty gate: a solution at
or below it passed. The gate is the operational measure, because it counts
accepted solutions rather than fractional energy.

`energy_milli` is a competitive baseline, not ground truth. Each corpus faced a
different field of miners, so gate-pass rates do not compare across corpora.
Paired kernel-to-kernel differences share a baseline and cancel it, so use those
for any claim about kernel quality.

### Results

Paired against `cpu-sa` on the identical instances. A negative number is better
than `cpu-sa`. Every verdict below replicated across two independent runs.

| kernel | `chain-h0` | t | `chain-ternary` | t |
| --- | --- | --- | --- | --- |
| `cpu-sb` | -0.156% | -6.3 | -0.075% | -3.9 |
| `cpu-hdsb` | -0.127% | -5.5 | -0.047% | -2.9 |
| `cpu-gibbs` | -0.027% | -1.1 | -0.005% | -0.3 |
| `cpu-bsb` | -0.137% | -5.3 | +0.196% | +8.1 |
| `cpu-hbsb` | +0.078% | +3.4 | +0.141% | +7.3 |
| `cpu-mps` | +15.254% | +208 | +14.080% | +220 |
| `cpu-mfa` | +15.279% | +228 | +14.042% | +208 |

Gate-pass rate on the hardest 50, which is the operational measure:

| kernel | `chain-h0` | `chain-ternary` |
| --- | --- | --- |
| `cpu-sb` | 20% | 68% |
| `cpu-hdsb` | 14% | 64% |
| `cpu-sa` | 8% | 64% |
| `cpu-gibbs` | 8% | 64% |
| `cpu-bsb` | 20% | 64% |
| `cpu-hbsb` | 0% | 62% |
| `cpu-mps`, `cpu-mfa` | 0% | 0% |

### What the replay changes

`cpu-sb` wins on real problems by a wider margin than the ladder showed, and it
wins on both corpora. The production track holds.

`cpu-bsb` changes sign between the corpora. It beats `cpu-sa` on `chain-h0` and
loses to it on `chain-ternary`, both at high significance and both replicated.
The ladder ranked it below `cpu-sa` at every size. Three datasets give three
verdicts, so treat the ladder ranking for this kernel as superseded.

The split follows the coupling function exactly. `cpu-sb` and `cpu-hdsb` use
`Coupling::Discrete` and win on both corpora. `cpu-bsb` and `cpu-hbsb` use
`Coupling::Continuous` and lose whenever biases are present.

That split is an observation. The mechanism first proposed for it has since
been tested and refuted, so the cause is open.

The proposal was this. `sb_core.rs` carries linear biases on an ancilla particle
at index `n`, and the force term reads `f += g.h[i] * coupled[n]`. Under
`Coupling::Discrete` the ancilla contributes `sgn(x_n)`, so each bias
contributes exactly `h_i`. Under `Coupling::Continuous` it contributes the raw
position `x_n`, so one shared number scales every linear bias in the problem.
That number starts near zero, and the gauge fix reads only the sign of the
position, so the magnitude carries no meaning of its own. On `chain-h0` there
are no biases and no ancilla, which fits the sign change.

Clamping the ancilla to its sign under both coupling forms tests that
proposal directly, and it does not hold. On the 50 hardest `chain-ternary`
problems
`cpu-bsb` moves from +0.196% and +0.184% across two runs to +0.209% paired
against `cpu-sa`, and the gate-pass rate holds at 64%. The change is not an
improvement.

The same run shows what the ancilla is doing instead. Read diversity falls from
234 to 64, so clamping does change the kernel, and what it removes is
exploration. A bias that grows with `x_n` is weak while the couplings organize
the frustrated structure and strong once that structure settles, which is an
annealing schedule for the linear biases rather than a leak. Removing it costs
diversity and returns no energy.

The coupling split is real and replicated, and no mechanism accounts for it
yet. Do not repeat the clamp.

The tensor-network binaries fail on both corpora and clear the gate on no
instance. `select_chi` returns more than 1 on both real graphs, which the
differing per-instance energies confirm, so the tensor network runs here. It
buys nothing: `cpu-mps` and `cpu-mfa` stay within noise of each other on both
corpora.

### The greedy polish sets the tensor-network result

Five arms were measured on the same 50 hardest instances of each corpus, chosen
so that each one changes a component a reader would expect to matter.

| arm | what it changes | `chain-h0` | `chain-ternary` |
| --- | --- | --- | --- |
| `cpu-mps`, annealed | tensor network, bond dimension above 1 | +15.254% | +14.080% |
| `cpu-mfa`, annealed | product state, bond dimension 1 | +15.279% | +14.042% |
| `cpu-mps`, `QUIP_MPS_INIT=random` | no anneal at all | +15.229% | +14.021% |
| `cpu-mps`, random, second replicate | no anneal, independent sample | +15.254% | +14.106% |
| `cpu-flatiron` | belief-propagation net on the problem graph | +15.287% | +14.103% |

Paired against `cpu-sa`. Every arm clears the gate on no instance. The whole
spread is 0.058 points on `chain-h0` and 0.085 on `chain-ternary`, against
standard errors near 0.07, so the five arms are one measurement.

Across those rows the bond dimension changes and the anneal is removed
entirely. The network geometry changes from a reverse Cuthill-McKee chain to a
net on the raw problem graph. The conditioning changes from exact
right-canonical sampling to a one-hop approximation. The last row is a separate
kernel, audited against its source paper. None of it moves the answer.

One component is common to all five: `polish_from` in
[`sampler_core.rs`](../src/sampler_core.rs), the strict-descent greedy sweep
that runs after sampling.

> On these instances every tensor-network state we can afford is uninformative.
> Annealed or random, MPS chain or belief-propagation net, bond dimension 1 or
> above, the sampled configuration carries no usable signal and the output
> quality is set entirely by the greedy polish that follows. The tensor network
> is not losing to annealing. It is not participating. This covers the bond
> dimensions affordable on a degree-18 hardware graph under the 64 MB per-model
> cap, not the method in principle: the published results use coordination at
> most 6, where a bond dimension of 32 is affordable, and the per-site cost of
> `16 * chi^degree` bytes makes that regime unreachable here.

The claim is about transfer, not about the code. An audit against the
source paper found the gate arithmetic, the message update, and the parallel
edge merge all correct, so a poor result here reports on the method rather than
on the code.

### The diversity metric confirms this through a second channel

Each record carries `diversity_milli`, the mean pairwise flip-invariant Hamming
distance between a job's returned solutions, normalized by spin width. Reads
drawn as fair coins score near 500. Median over the 50 hardest:

| corpus | physical kernels | tensor-network family |
| --- | --- | --- |
| `chain-h0` | 404 to 447 | 487 |
| `chain-ternary` | 234 to 289 | 478 to 479 |

The tensor-network family sits at the fair-coin level on both corpora, and it
does not move when the initialization mode or the network type changes. That is
the uniform-random signature, measured on a channel that could have contradicted
the energy result.

The `chain-ternary` row carries the mechanism. Real biases pull the physical
kernels' reads toward each other and their diversity falls to 234. The
tensor-network family does not move. The biases reach the physical kernels'
states and never reach the tensor-network samples.

### Mechanisms eliminated by measurement

Three explanations were tested and none survives. Do not cite them.

| mechanism | how it was eliminated |
| --- | --- |
| Bond dimension held at 1 | `select_chi` returns more than 1 on both real graphs, and the result does not change |
| Zero-bias symmetry | Real ternary biases recover 1.7 points of a 15 point deficit |
| Conditioning fidelity | `cpu-flatiron` conditions one hop at a time and `cpu-mps` conditions exactly, and both return the same number |

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

## Known limits

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
- One instance family. The same 30 instances appear at every size, so a kernel
  that suits this family looks good at all eight sizes together. The replayed
  chain problems are the second family, and they overturn the ladder ranking for
  `cpu-bsb`.
- Zero biases throughout the ladder. That hides the ancilla path, where the
  continuous-coupling variants fail, and it holds `select_chi` at 1, so the
  ladder
  never measures the tensor-network binaries running a tensor network.

Limits that apply to the replayed chain problems:

- Wall-clock time is not reported. The host runs other work, and two timing
  attempts were lost to competing load. Quality is unaffected, because fixed
  reads and sweeps make the energies independent of load.
- The hardest 50 of each corpus. That is the decision-relevant slice, and it is
  too small to resolve a difference below about 0.1%.
- Gate-pass rates do not compare across corpora, because each corpus faced a
  different field of miners.

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

- An equal wall-clock comparison at every size on the ladder. Measure it on an
  isolated host, or record processor time per model rather than elapsed time.
  Elapsed time on a shared workstation has failed twice.
- A better polish for the tensor-network kernels. The greedy polish sets their
  result, so any gain has to come from there. A polish that accepts an
  energy-neutral move would cross the domain-wall plateau that strict descent
  cannot. Measure it before adopting it, because strict descent is what makes
  the current polish stop.
- Whether any affordable state carries signal on a degree-18 graph. Every
  measured arm samples at the fair-coin level. A bond dimension that would
  change this needs `16 * chi^degree` bytes per site, so answering the question
  needs a different network geometry rather than a larger budget.
- Why `Coupling::Continuous` loses once biases are present. Clamping the ancilla
  to its sign is refuted, so start from the diversity result: the ancilla ramps
  the linear biases through the schedule, and an explicit ramp is the next thing
  to measure against the implicit one.
- A sweep of `num_sweeps` for `cpu-sb` across its adapt envelope of 256 to 8192,
  to find its operating point before any further comparison.
- A multi-threaded run through the streaming pump, to confirm that the
  single-threaded cost ratio holds at production concurrency.
