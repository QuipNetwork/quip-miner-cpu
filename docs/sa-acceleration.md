# Accelerating the simulated annealing kernel

This document records which techniques from Isakov, Zintchenko, Rønnow and
Troyer, *Optimized simulated annealing for Ising spin glasses* (Comput. Phys.
Commun. **192**, 265 (2015), [arXiv:1401.1084](https://arxiv.org/abs/1401.1084))
apply to this miner. It states which ones this crate now ships, and what they
measure.

## Result

On the two bundled Zephyr corpora, at equal sweep and read counts:

| kernel | jobs per minute, 64 reads | speedup | quality change |
| --- | --- | --- | --- |
| `cpu-sa` before this work | 94.1 | 1.00x | baseline |
| `cpu-sa` after this work | 110.8 | 1.18x | bit-identical |
| `cpu-fsa` | 137.7 | 1.46x | none measurable |
| `cpu-msa` | 958.7 | 10.19x | none measurable |

Measured on all 500 instances of the isingmark hardest-500 `chain-h0` corpus,
with 2 repetitions per arm. Energy quality holds: every paired difference stays
under 0.005 percent at a `t` below 1. Diversity holds for `cpu-sa` and
`cpu-fsa`. For `cpu-msa` it falls 0.07 percent, which is below the unit the
protocol reports. The gain is one temperature ladder of identical work done
faster.

## Which techniques apply

The paper lists eight optimizations. Their applicability to a degree-20 Zephyr
graph with `J = ±1` differs from their applicability to the degree-6 Chimera
graph the authors targeted.

| # | technique | status here |
| --- | --- | --- |
| 1 | cache the effective field, update on flip | already present as `heff[]` |
| 2 | fuse the loop over neighbours | already present |
| 3 | reorder spins for locality | not done, the graph is already compact |
| 4 | precompute the acceptance table per temperature | **new**, `sa_int.rs` |
| 5 | log-transform the Metropolis test | **new**, `sa_int.rs` |
| 6 | fast random numbers | not done, see below |
| 7 | fixed-point energies | **new**, folded into technique 4 |
| 8 | multi-spin coding across replicas | **new**, `sa_msc.rs` |

### Nothing in the paper is exponential in maximum degree

This answers the concern that the tricks carry an exponential cost in the
maximum degree, and so work for Chimera but not for Advantage or
Advantage2.

The released reference code hard-codes a maximum degree of 6 because
Chimera has degree 5 to 6. That is a property of the code the authors shipped,
not of the method. Every cost in the paper is linear:

- The acceptance table has one entry per reachable value of the local field.
  Its size is the product of the degree and the coupling range, so it grows
  linearly. At degree 20 with unit couplings the table holds 101 entries per
  temperature rung.
- Multi-spin coding counts satisfied bonds in `⌈log2 k⌉` bit planes for a
  degree of `k`. Degree 20 needs 5 planes. Degree 63 needs 6.
- The carry-save adder that fills those planes costs one pass per incident
  bond, so it is linear in the degree.

Degree 20 sits inside the method, not outside it.

The real cost of degree 20 appears somewhere else. Multi-spin coding cannot
keep a cached effective field, because each of the 64 replicas accepts a
different set of flips. Every word update recomputes the field from the
neighbours. At degree 6 that recomputation is cheap. At degree 20 it is most of
the cost of the kernel, and it is the reason the measured kernel gain is 17.5x
rather than the 64x that one word of parallel work suggests. The kernel-rate
table below makes this visible: on a graph small enough to sit in L2 the same
kernel reaches 40x.

### Techniques specific to `J = ±1`

Two of the gains depend on the coupling values, as the source suggests.

With integer couplings the energy change of one flip is `ΔE = 2n` for a small
integer `n`. The Metropolis test `u < exp(-βΔE)` then becomes a comparison
against a table indexed by `n`. That removes the `exp()` call.

The log transform goes further. Rewrite the test as `ΔE < -(1/β) ln u`. With
`a = exp(-2β)` the acceptance probability is `a^n`, so a variate `M` drawn with
`P(M ≥ n) = a^n` is geometric, and accepting when `n ≤ M` is exact. Because a
downhill or flat candidate has `n ≤ 0 ≤ M`, the whole Metropolis test collapses
to one integer comparison, `m ≥ -M`. The kernel draws no random number per
candidate flip.

This crate draws the thresholds once per job into a table shared by every read,
and shifts the read position by a random amount each sweep, so that no site
reads the same threshold twice in a row.

The field-free case, `h = 0`, buys nothing extra here. A field in `{-1, 0, +1}`
folds into the same kernel as one extra bond to a spin pinned at `+1`, so
`chain-h0` and `chain-ternary` run the same code. The measured gains are within
noise of each other on the two corpora, which confirms it.

## What shipped

Three kernels, in `src/sa_int.rs` and `src/sa_msc.rs`.

**The exact integer kernel.** `IntGraph` holds fields as `i8` and stores the
coupling sign in the top bit of each neighbour identifier, so both fit one
`u32`. A sweep streams 4 bytes
per incident edge instead of 12. On the bundled corpora that is 332 KiB of
neighbour data per sweep instead of 996 KiB. Acceptance probabilities come from
a table built once per temperature rung. `cpu-sa` uses this kernel whenever the
problem qualifies, and falls back to the `f64` kernel when it does not.

**The tabulated-threshold kernel**, exposed as `cpu-fsa`. Same representation,
with the per-candidate random draw replaced by the geometric threshold table
described earlier.

**The multi-spin coded kernel**, exposed as `cpu-msa`. One `u64` holds the same
lattice spin across 64 replicas. A replica is a read, so a 64-read job is one
word pass. Satisfied bonds accumulate in 6 bit planes through a ripple
carry-save adder, and the acceptance test is a branchless comparison of that
counter against a scalar.

## Why `cpu-sa` itself changed

The integer kernel is bit-identical to the `f64` kernel it replaces, so it went
into the production binary rather than behind a new identity.

Under the preconditions the integer path enforces, the `f64` kernel's effective
field is already an exact integer at every step. It is seeded from a sum of `±1`
and `0` terms, and each accepted flip adds exactly `±2`. Every such value is
exact in `f64`. That makes the `f64` kernel always evaluate `exp((-ΔE) * β)`
with `-ΔE = -2n`, while the table evaluates `exp((-2β) * n)`. The product `-2β`
is exact, because it is a sign flip and an exponent decrement. Both expressions
are a single rounding of the same real number `-2βn`, so both produce the same
`f64`. Given the same random stream, both kernels make the same accept and
reject decisions.

`differential_matches_the_f64_kernel_on_unit_couplings` pins that claim. It runs
both kernels from the same seed and requires identical spins.

`cpu-fsa` and `cpu-msa` are not bit-identical, because the log transform changes
which random numbers are drawn. They ship as new identities.

## Preconditions and fallback

`IntGraph::from_base` returns `None`, and the caller stays on the `f64` kernel,
unless:

- every kept coupling is exactly `+1`, `-1`, or `0`, and
- every field is a whole number, and
- `maxᵢ (|hᵢ| + degᵢ) ≤ 100`.

One more condition bounds memory rather than correctness. The acceptance table
holds one row per temperature rung. For an unbounded sweep count that reaches
101 times the memory the `f64` kernel needs for the same ladder. Past 2^20 table
entries the integer path declines, which leaves the caller on the `f64` kernel,
whose spins are the same. A mining job runs at most 1024 rungs, so
nothing in the miner approaches the bound.

`cpu-fsa` and `cpu-msa` draw a threshold row of 8 KiB per rung, and decline past
64 MiB of them, which falls back to `cpu-sa`. A problem with more than 8192
nodes reuses a threshold within one sweep, for sites 8192 apart. The per-sweep
offset repairs the pairing every sweep, so no two sites share a threshold twice,
but the count of independent thresholds in one sweep is capped there.

`cpu-msa` needs two more conditions, and falls back to the tabulated kernel
otherwise:

- every field is in `{-1, 0, +1}`, so it folds into one ghost bond, and
- the effective degree, including that ghost bond, is at most 63.

Both bundled corpora meet these conditions. `chain-h0` has `h = 0` and
`J = ±1`.
`chain-ternary` has `h ∈ {-1, 0, 1}` and `J = ±1`. Maximum degree is 20.

## Measurements

### Kernel rates

Spin updates per second on the real Zephyr adjacency, single-threaded, from
`docs/comparisons/sakernels.rs`:

| graph | `f64` | integer | tabulated | multi-spin |
| --- | --- | --- | --- | --- |
| Zephyr, n=4577 | 74 M/s | 90 M/s (1.21x) | 112 M/s (1.51x) | 1299 M/s (17.5x) |
| random, n=4577 | 72 M/s | 88 M/s (1.23x) | 107 M/s (1.50x) | 1253 M/s (17.5x) |
| random, n=512 | 83 M/s | 102 M/s (1.23x) | 141 M/s (1.71x) | 3317 M/s (40.2x) |

The multi-spin rate counts all 64 lanes. A job that asks for 36 reads leaves 28
lanes idle, so the same kernel gives that job 9.9x rather than 17.5x.

The n=512 row shows what the degree-20 field recomputation costs. That graph
fits in L2, and the multi-spin kernel reaches 40x there against 17.5x on
Zephyr.

### End-to-end, isingmark hardest-500 `chain-h0`

The primary campaign. All 500 instances of
`isingmark/subset500/hardest500_h0.jsonl`, 2 repetitions, 5 arms, both read
counts: 10000 jobs. `sa-base` is the `cpu-sa` binary from `origin/main`, before
this change.

Hardness 0.5 gives 36 reads and 550 sweeps. Hardness 1.0 gives 64 reads and 1000
sweeps, which is exactly one `u64` word.

The corpus topology and the `chain-h0` preset are the same graph. Canonicalised
(sorted keys, compact separators) the two files hash identically, with matching
node order and edge order, not merely matching counts.

#### Hardness 1.0, 64 reads

| kernel | mean gap | vs `sa-base` | gate | median wall | jobs/min | speedup | diversity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sa-base` | +1.641% | — | 10% | 4916 ms | 94.1 | 1.00x | 443.40 |
| `cpu-sa` | +1.641% | -0.001% (t -0.1) | 11% | 4174 ms | 110.8 | 1.18x | 443.44 |
| `cpu-fsa` | +1.642% | +0.001% (t +0.2) | 10% | 3337 ms | 137.7 | 1.46x | 443.39 |
| `cpu-msa` | +1.645% | +0.003% (t +0.8) | 10% | 380 ms | 958.7 | 10.19x | 443.13 |
| `cpu-sb` | +1.565% | -0.076% (t -19.3) | 16% | 5744 ms | 80.9 | 0.86x | 437 |

#### Hardness 0.5, 36 reads

| kernel | mean gap | vs `sa-base` | gate | median wall | jobs/min | speedup | diversity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sa-base` | +1.941% | — | 1% | 1560 ms | 288.0 | 1.00x | 450.14 |
| `cpu-sa` | +1.945% | +0.004% (t +0.8) | 0% | 1328 ms | 337.5 | 1.17x | 450.14 |
| `cpu-fsa` | +1.941% | +0.000% (t +0.1) | 1% | 1067 ms | 414.4 | 1.44x | 450.10 |
| `cpu-msa` | +1.939% | -0.002% (t -0.3) | 1% | 213 ms | 1623.0 | 5.64x | 449.80 |
| `cpu-sb` | +1.823% | -0.118% (t -24.3) | 4% | 1820 ms | 246.9 | 0.86x | 445 |

#### What the mean gap does and does not say

On 494 of the 500 instances, every arm finished above the energy the corpus
records for that nonce. The gap measures distance below a target the chain
already beat. Its absolute size says more about the corpus than about any
kernel. It is still the right quantity for the
paired columns, because every arm faces the same target on the same instance.

### End-to-end, hardest-50 subset

An earlier campaign on the 50 hardest instances of each corpus, 3 repetitions,
arms interleaved within each repetition. Kept because it covers `chain-ternary`,
which has no 500-instance corpus, and because it is the only campaign with
3 repetitions.

Those 50 nonces are a subset of the 500. Restricting the 500-instance run to
them reproduces the separate campaign: +0.377% against +0.387% for `cpu-sa`.
The full 500 shows a much larger mean gap than its own hardest-50 subset,
because "hardest" ranks by how deep the chain went, not by how hard the instance
is for these kernels.

#### Hardness 0.5, `chain-h0`

| kernel | mean gap | vs `sa-base` | gate | median wall | jobs/min | speedup | diversity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sa-base` | +0.701% | — | 7% | 1392 ms | 294.7 | 1.00x | 446 |
| `cpu-sa` | +0.682% | -0.019% (t -1.4) | 8% | 1149 ms | 352.7 | 1.20x | 445 |
| `cpu-fsa` | +0.704% | +0.003% (t +0.2) | 7% | 926 ms | 431.8 | 1.47x | 446 |
| `cpu-msa` | +0.711% | +0.011% (t +0.8) | 11% | 196 ms | 1576.2 | 5.35x | 445 |
| `cpu-sb` | +0.589% | -0.112% (t -8.3) | 15% | 1628 ms | 253.3 | 0.86x | 440 |

#### Hardness 0.5, `chain-ternary`

| kernel | mean gap | vs `sa-base` | gate | median wall | jobs/min | speedup | diversity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sa-base` | +0.106% | — | 65% | 1405 ms | 291.0 | 1.00x | 286 |
| `cpu-sa` | +0.094% | -0.013% (t -1.1) | 65% | 1170 ms | 345.8 | 1.19x | 286 |
| `cpu-fsa` | +0.099% | -0.007% (t -0.8) | 65% | 933 ms | 429.5 | 1.48x | 286 |
| `cpu-msa` | +0.098% | -0.008% (t -0.8) | 64% | 204 ms | 1539.6 | 5.29x | 285 |
| `cpu-sb` | +0.032% | -0.074% (t -8.3) | 65% | 1727 ms | 239.7 | 0.82x | 272 |

#### Hardness 1.0, `chain-h0`

| kernel | mean gap | vs `sa-base` | gate | median wall | jobs/min | speedup | diversity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sa-base` | +0.390% | — | 45% | 4620 ms | 91.6 | 1.00x | 438 |
| `cpu-sa` | +0.387% | -0.003% (t -0.3) | 49% | 3816 ms | 110.1 | 1.20x | 438 |
| `cpu-fsa` | +0.391% | +0.002% (t +0.1) | 51% | 2988 ms | 139.2 | 1.52x | 438 |
| `cpu-msa` | +0.391% | +0.002% (t +0.2) | 48% | 363 ms | 915.7 | 10.00x | 438 |
| `cpu-sb` | +0.313% | -0.077% (t -6.7) | 61% | 5336 ms | 79.1 | 0.86x | 432 |

#### Hardness 1.0, `chain-ternary`

| kernel | mean gap | vs `sa-base` | gate | median wall | jobs/min | speedup | diversity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sa-base` | -0.125% | — | 71% | 4658 ms | 85.8 | 1.00x | 261 |
| `cpu-sa` | -0.124% | +0.001% (t +0.1) | 71% | 3822 ms | 102.2 | 1.19x | 260 |
| `cpu-fsa` | -0.124% | +0.001% (t +0.2) | 73% | 3007 ms | 137.1 | 1.60x | 260 |
| `cpu-msa` | -0.133% | -0.008% (t -1.0) | 72% | 387 ms | 840.6 | 9.80x | 258 |
| `cpu-sb` | -0.168% | -0.043% (t -6.0) | 74% | 5674 ms | 71.9 | 0.84x | 241 |

### How to read these tables

`mean gap` is the mean relative distance from the chain-recorded best energy,
where a negative number means the kernel beat the recorded chain. `vs sa-base`
pairs each kernel against the pre-change baseline by job identifier, so it
removes instance-to-instance variation. The `t` value is the paired mean divided
by its standard error. `gate` is the fraction of runs that met the corpus
difficulty gate. `diversity` is the protocol diversity score in milli-units.

Every `t` value for the three annealing kernels is below 1.5 in absolute value
on 500 paired instances. Energy quality is unchanged.

`cpu-sb` is the one arm whose energy differs, and the difference is an
improvement: 0.076% and 0.118% deeper, which lifts the gate rate from 10 to 16
percent and from 1 to 4 percent, at 0.86x throughput. It is a Simulated
Bifurcation kernel, included here for reference.

**Diversity moves slightly for `cpu-msa`, and the earlier campaign was too small
to see it.** Against `cpu-sa` the multi-spin kernel loses 0.31 milli at hardness
1.0 and 0.34 milli at hardness 0.5, which is t = +2.4 and t = +2.8 over 1000
paired records. The effect is real. It is also 0.07 percent, below the whole
milli-unit the protocol reports.

The mechanism is the lane coupling the design predicted. Replicas share one
acceptance threshold per word update, so two lanes that ever reach the same
configuration stay identical for the rest of the run. Fifty instances put this
inside the noise, while five hundred resolve it. Nothing in the shipped code
changes on account of 0.07 percent, but the earlier claim that the coupling is
unmeasurable was wrong. A campaign that raises read counts or lowers temperature
should measure it again.

### Why 5.3x and 10.0x

Two effects separate the two settings, and both are visible in the numbers.

**Lane occupancy separates 5.3x from 10.0x.** `cpu-msa` fills 64 lanes per word
pass. A 64-read job uses all of them. A 36-read job leaves 28 idle. The measured
ratio between the two settings is 10.00 divided by 5.35, which is 1.87, against
the 64 divided by 36 that lane counting predicts, which is 1.78.

**Work outside the annealing loop caps both.** Each setting reaches a little
over half of its kernel-level figure: 5.35 of 9.9, and 10.00 of 17.5. Building
the graph, scoring every read, and computing the diversity score are the same
cost for every kernel, and the multi-spin kernel does not make them faster. Once
annealing stops being most of a job, further kernel speedup stops showing up in
jobs per minute.

The first effect makes the read count the control that matters for `cpu-msa`.
Read counts that are multiples of 64 waste no lanes. The second effect sets the
ceiling, and lifting it means making scoring faster, not annealing.

## What was considered and not built

**Fast random numbers, technique 6.** The paper replaces its generator with a
linear congruential generator. This crate keeps `xoshiro`. The log transform
already removes 63 of every 64 draws in the multi-spin kernel and every
per-attempt draw in `cpu-fsa`, so the generator is no longer on the hot path.
Trading generator quality for speed that is no longer needed is a bad trade for
a miner scored on diversity.

**Splitting positive and negative couplings into separate rows.** This would
remove the per-edge sign materialisation from the multi-spin inner loop. It
costs a second CSR structure and complicates the neighbour walk. Measure before
building it.

**Holding one replica out of each update to decorrelate the lanes.** The
multi-spin kernel shares one acceptance threshold across the 64 replicas of a
word update, which couples them. The coupling has a sharp form: two lanes that
ever reach the same configuration make the same decision at every later site,
so they stay identical for the rest of the run. Coalescence is absorbing.
Because the miner is scored on diversity, this was the main risk in the design.

The 500-instance campaign resolves the effect and sizes it. `cpu-msa` loses
0.31 milli of diversity at hardness 1.0 and 0.34 at hardness 0.5 against
`cpu-sa`, at `t` of 2.4 and 2.8. That is 0.07 percent, below the whole
milli-unit the protocol reports, and it costs nothing at the read counts and
temperatures measured here. The decorrelation machinery stays on paper on those
grounds. The effect is present, merely small. Build the machinery for a
campaign that raises the read count, or one that moves to a landscape which
funnels replicas into a single basin.

**Spin reordering, technique 3.** The Zephyr fixture already relabels to a dense
range in ascending identifier order, and the neighbour rows are contiguous. No
measurement suggests a reordering pass would beat that layout.

## Reproducing

Build with the `experimental` feature to get `cpu-fsa` and `cpu-msa`:

    cargo build --release --all-features

The kernel-rate program is a loose file, following the convention of the other
programs in `docs/comparisons/`. To run it, copy it into `src/bin/`, widen
`sa_int` and `sa_msc` to `pub mod` in `src/lib.rs`, widen their `pub(crate)`
items to `pub`, move `serde_json` from `[dev-dependencies]` to
`[dependencies]`, and point `QUIP_TOPOLOGY_SPEC` at a topology fixture:

    cp docs/comparisons/sakernels.rs src/bin/
    QUIP_TOPOLOGY_SPEC=../isingmark/fixtures/chain-h0.spec.json \
      cargo run --release --all-features --bin sakernels

Revert all four edits afterward. The crate keeps those items private.

The end-to-end tables come from the isingmark `throughput` harness. See
`docs/comparisons.md` for the method.

## Reference

Sergei V. Isakov, Ilia N. Zintchenko, Troels F. Rønnow, Matthias Troyer.
*Optimized simulated annealing for Ising spin glasses.* Computer Physics
Communications **192**, 265-271 (2015). arXiv:1401.1084.
