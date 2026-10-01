# Round 2 regime search report

Status: draft, campaign data, incomplete. This report draws on `campaign` CPU records under `<CHECKOUT_ROOT>/research/round2/tests/fixtures/empty-cpu-root`. No Round 2 QPU capture is loaded for this report run (pass --qpu-root and --physical-capture-manifest to load one). This draft states no regime verdict.

## Model identity

| Cell | Distinct models observed | Distinct model hashes |
| -- | -- | -- |
| `native-pm1` | 0 | 0 |

## Solver settings

| Cell | Kernels observed | Sweep depths observed | Reads |
| -- | -- | -- | -- |
| `native-pm1` | n/a | n/a | n/a |

## Depth, quality, and time

Every row is one (cell, kernel, sweep-depth) arm. Energy is the primary comparison. Timing uses `timing-subset` records from a loaded host with parallel workers, one per physical core. For each job, its fastest successful run is its run speed. The timing columns show the median sampling and wall times across successful, supported records, regardless of host contamination or timing mode. The contaminated and parallel wall-time columns remain separate diagnostics. Best energy uses every successful record. Best energy remains from campaign records. Sampling time is `elapsed_sampling_s`. Wall time is end-to-end.

| Cell | Kernel | Sweeps | Observed | Missing | Completed | Failed | Unsupported | Nonfinite | Median best energy | Sampling s, fastest run from timing-subset (n) | Wall s, fastest run from timing-subset (n) | Wall s, contaminated (n) | Wall s, parallel (n) |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| `native-pm1` | `cpu-sa` | 512 | 0 | 100 | 0 | 0 | 0 | 0 | n/a | n/a (0; no matching timing-subset record) | n/a (0; no matching timing-subset record) | n/a (0) | n/a (0) |

## Lane diversity by arm

Unique states and pairwise Hamming distances come from each saved `.npz` spin batch. Per-arm unique counts and mean distances are averages across records. The minimum is the lowest within-record pairwise distance. Seeded-sweep-v2 separates the saved seeded and cold lane batches.

| Cell | Kernel / seed source | Sweeps | Lane group | Records | Mean unique states / record | Mean pairwise Hamming / record | Minimum pairwise Hamming |
| -- | -- | -- | -- | -- | -- | -- | -- |
| n/a | n/a | n/a | n/a | 0 | n/a | n/a | n/a |

## Paired CPU kernel gaps

Each row pairs one kernel against `cpu-sa` on the same models (matched by `model_hash`), never by comparing medians over two arms' completed sets, which can differ. Compared counts only models where both kernels completed with a finite energy at that depth.

| Cell | Kernels | Sweeps | Compared | Strict other/base/tie |
| -- | -- | -- | -- | -- |

No comparable pairs against `cpu-sa` for any kernel or depth given.

## Quantum processing unit wins, ties, and losses

Every row pairs one physical-scale QPU arm (cell, scale, anneal time) against one CPU arm (kernel, sweep depth), matched by `model_hash` (review, C1). Compared counts only models with both a real, manifest-verified capture and a completed, finite CPU record.

| Cell | Scale | Anneal us | Kernel | Sweeps | Compared | Strict qpu/cpu/tie | Numeric qpu/cpu/tie | Material qpu/cpu/tie |
| -- | -- | -- | -- | -- | -- | -- | -- | -- |

No comparable pairs exist because no cell has both a physical-pilot QPU capture and a matching CPU arm yet.

## Paired gaps

Bootstrap paired-gap intervals (`round2_metrics.bootstrap_paired_gap`: 10,000 resamples, the fixed recorded seed, grouped by model, since every physical-pilot model is an independent draw). The gap is the same signed relative gap `quality_outcome` itself computes: negative means the QPU energy is lower (better).

| Cell | Scale | Anneal us | Kernel | Sweeps | Groups | Point | 95% low | 95% high | Descriptive |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |

No comparable pairs exist because no cell has both a physical-pilot QPU capture and a matching CPU arm yet.

## QPU against CPU at matched run time

QPU time is charged access time for 64 reads, with end-to-end time also shown. CPU time is sampling time for 64 reads from the campaign, using the fastest successful attempt on a loaded host with parallel workers. Energy outcomes use the same strict, numeric-tolerance, and material rules as the equal-sweep tables. Equal budget uses the deepest CPU depth completed within that capture's access-time budget. Time to QPU energy uses the quickest CPU depth that reached the capture's best energy.

| Cell | Scale | Anneal us | Kernel | Models | Median QPU access s | Median QPU end-to-end s | Equal budget: strict / numeric / material qpu/cpu/tie | Over budget | Reached QPU energy (n of Models) | Not reached | Median CPU s to QPU energy | Median ratio to QPU access |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| n/a | n/a | n/a | n/a | 0 | n/a | n/a | n/a | 0 | 0 of 0 | 0 | n/a | n/a |

No captured QPU arms are available for a matched-run-time comparison.

## Physical scale

The physical-range pilot (12 sorted nonces, 3 scales, 2 anneal times -- the design doc's initial campaign proposal) prices captures before they run. A cell with no arm here has no physical-scale plan at all, not merely no result yet.

| Cell | Anneal times planned (µs) | Captures planned | Reads per capture | Status |
| -- | -- | -- | -- | -- |
| `native-pm1` | n/a | n/a | n/a | unavailable -- no physical-scale plan for this cell |

## Historical context (cited, not recomputed here)

The current portfolio manuscript reports 1,930 comparable races, with 833 QPU quality wins and 1,097 ties at 0.5% materiality. At zero tolerance it reports 970 QPU wins and 960 ties, and neither table contains an SA win. The manuscript reports about 70% faster device access but supplies no joint quality/time counts, so these figures do not establish an 80% strict-quality win rate (design doc, `docs/superpowers/specs/2026-09-22-regime-search-round2-design.md`).
The older 2,088-race window overlaps the manuscript's 1,930-race window. This report keeps the two separate and never pools them, because a union of overlapping windows would double count shared races.
The 80% portfolio claim stays unresolved. No source used in this report defines that metric together with the denominator or joint counts a claim at that scale would need.

## What each regime establishes, and what runs next

| Regime | Established | Unresolved | Next control |
| -- | -- | -- | -- |
| `native-pm1` | 0 completed CPU timing/quality records across 0 kernels (none yet) from the campaign run. | No Round 2 QPU capture exists yet, and this cell has no physical-scale plan at all. The CPU comparison alone cannot answer the regime question. | The CPU campaign for this cell is complete. Decide whether a QPU arm belongs in this regime's next round. |
| Portfolio pipeline | The deadline arm (synthetic fixtures, historical settings) ran to completion for both baskets and both beta labels, producing repaired objectives and raw feasibility counts. | No paired QPU portfolio result exists, so strict and material wins, speed-only outcomes, and joint quality/time counts stay unavailable. The 80% portfolio claim stays unresolved. | Capture the portfolio pilot (12 frozen market instances, beta-zero and positive-beta controls) once its provenance and anneal-setting classification are explicit, per the design doc. |
