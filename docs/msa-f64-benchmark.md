# Float multi-spin annealing benchmark

This report compares float64 multi-spin annealing with scalar simulated annealing. Times use Coordinated Universal Time (UTC).

## Host and run

| Item | Recorded value |
| --- | --- |
| CPU | Apple M4 Max |
| Logical CPU count | 16 |
| Memory | 128 GiB, or `137438953472` bytes |
| Operating system | macOS 26.5.2, build 25F84 |
| Start, UTC | 2026-10-11 22:50:49 |
| End, UTC | 2026-10-12 00:01:20 |
| Elapsed time | 4,231 seconds, or 1 hour 10 minutes 31 seconds |
| Uptime before | `18:50 up 32 days, 7:59, 8 users, load averages: 3.45 4.22 4.82` |
| Uptime after | `20:01 up 32 days, 9:10, 8 users, load averages: 3.62 3.64 3.81` |

The host facts come from `sysctl` and `sw_vers`. The benchmark binary used base commit `fd0d170` plus the uncommitted three-sweep example. The example enters the same commit as this report.

## Method

The controller ran `./target/release/examples/msa_research_bench --models 5 --repeats 3` after building the release example. The run produced 255 JSON rows. It used five models per regime. Model seeds were 0 through 4. Each call requested 64 reads, and each point had three timed repeats. The example used the default `SampleParams` values for all other parameters.

| Regime | Model | Kernels |
| --- | --- | --- |
| `native-pm1` | Ring lattice, 4,096 spins, degree 20, J = ±1, h = 0 | `cpu-sa`, `cpu-msa-unit`, `cpu-msa-f64` |
| `fractional` | Same ring topology, J and h uniform in ±[0.01, 1] | `cpu-sa`, `cpu-msa-f64` |
| `multilevel` | Same ring topology, J in ±{1..7}, h in ±{0..3} | `cpu-sa`, `cpu-msa-f64` |
| `clique-portfolio` | Complete graph, 108 spins. Both J and h are fractional. | `cpu-sa`, `cpu-msa-f64` |
| `cubic-pm1` | Periodic 13 × 13 × 13 cubic lattice, J = ±1, h = 0 | `cpu-sa`, `cpu-msa-unit`, `cpu-msa-f64` |

Every kernel received the same graph for each model seed. The plan's five regimes differ from the Round 2 design appendix, so this run does not reproduce that campaign.

The measured sweep counts were 512, 2,048, and 8,192. The smoke run projected more than two hours with 32,768 sweeps. The controller dropped that count before the full run. Each call reports the median of three timings. The tables take the median of those five model timings. Best energy is the lowest energy among a model's 64 reads. The tables average that value and the mean read energy across models. They also average each model's distinct-state count. Lower energy is better.

A timed interval covers one `sample_research` call. Model construction and table generation are outside this boundary. The call includes its own validation, setup, annealing, scoring, and result creation. Each matched-time scalar budget comes from a one-repeat scalar probe at the float kernel's sweep count. The example uses the ratio of float time to probe time to calculate the scalar sweep count. It rounds that count before running scalar SA three times. The matched-time table reports the median scalar sweep count across five models, while probe time stays out of the result rows.

The time-to-quality target in this report is scalar SA's mean best energy at 8,192 sweeps, the largest measured count. A kernel reaches the target when its mean best energy is at or below that value. The table gives the least median call time among its measured sweep counts that reach the target.

### Matched sweeps

| Regime | Sweeps | Kernel | Median seconds | Mean best energy | Mean energy | Mean distinct |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `native-pm1` | 512 | `cpu-sa` | 0.930 | -13010.800 | -12950.650 | 64.0 |
| `native-pm1` | 512 | `cpu-msa-unit` | 0.084 | -13016.400 | -12951.900 | 64.0 |
| `native-pm1` | 512 | `cpu-msa-f64` | 1.287 | -13010.800 | -12951.550 | 64.0 |
| `native-pm1` | 2,048 | `cpu-sa` | 3.657 | -13056.400 | -13011.375 | 64.0 |
| `native-pm1` | 2,048 | `cpu-msa-unit` | 0.324 | -13062.800 | -13008.913 | 64.0 |
| `native-pm1` | 2,048 | `cpu-msa-f64` | 5.156 | -13058.000 | -13009.400 | 64.0 |
| `native-pm1` | 8,192 | `cpu-sa` | 14.586 | -13082.000 | -13044.400 | 64.0 |
| `native-pm1` | 8,192 | `cpu-msa-unit` | 1.272 | -13079.600 | -13045.163 | 64.0 |
| `native-pm1` | 8,192 | `cpu-msa-f64` | 20.650 | -13077.200 | -13043.475 | 64.0 |
| `fractional` | 512 | `cpu-sa` | 0.961 | -7891.843 | -7846.183 | 64.0 |
| `fractional` | 512 | `cpu-msa-f64` | 1.230 | -7891.729 | -7848.353 | 64.0 |
| `fractional` | 2,048 | `cpu-sa` | 3.721 | -7932.544 | -7903.610 | 64.0 |
| `fractional` | 2,048 | `cpu-msa-f64` | 4.896 | -7937.288 | -7903.334 | 64.0 |
| `fractional` | 8,192 | `cpu-sa` | 14.727 | -7958.445 | -7933.506 | 64.0 |
| `fractional` | 8,192 | `cpu-msa-f64` | 19.590 | -7956.842 | -7933.884 | 64.0 |
| `multilevel` | 512 | `cpu-sa` | 1.045 | -58767.600 | -58440.113 | 64.0 |
| `multilevel` | 512 | `cpu-msa-f64` | 1.275 | -58752.000 | -58449.188 | 64.0 |
| `multilevel` | 2,048 | `cpu-sa` | 4.124 | -59122.000 | -58866.725 | 64.0 |
| `multilevel` | 2,048 | `cpu-msa-f64` | 5.086 | -59125.200 | -58853.925 | 64.0 |
| `multilevel` | 8,192 | `cpu-sa` | 16.471 | -59343.600 | -59112.419 | 64.0 |
| `multilevel` | 8,192 | `cpu-msa-f64` | 20.229 | -59322.800 | -59098.350 | 64.0 |
| `clique-portfolio` | 512 | `cpu-sa` | 0.037 | -477.850 | -467.275 | 29.0 |
| `clique-portfolio` | 512 | `cpu-msa-f64` | 0.069 | -477.850 | -468.809 | 26.0 |
| `clique-portfolio` | 2,048 | `cpu-sa` | 0.139 | -477.850 | -471.799 | 16.0 |
| `clique-portfolio` | 2,048 | `cpu-msa-f64` | 0.277 | -477.850 | -471.414 | 16.8 |
| `clique-portfolio` | 8,192 | `cpu-sa` | 0.552 | -477.850 | -473.697 | 10.8 |
| `clique-portfolio` | 8,192 | `cpu-msa-f64` | 1.108 | -477.850 | -473.976 | 11.0 |
| `cubic-pm1` | 512 | `cpu-sa` | 0.455 | -3895.800 | -3863.275 | 64.0 |
| `cubic-pm1` | 512 | `cpu-msa-unit` | 0.051 | -3896.600 | -3863.575 | 64.0 |
| `cubic-pm1` | 512 | `cpu-msa-f64` | 0.556 | -3895.000 | -3864.438 | 64.0 |
| `cubic-pm1` | 2,048 | `cpu-sa` | 1.805 | -3908.600 | -3884.637 | 64.0 |
| `cubic-pm1` | 2,048 | `cpu-msa-unit` | 0.202 | -3908.600 | -3885.188 | 64.0 |
| `cubic-pm1` | 2,048 | `cpu-msa-f64` | 2.222 | -3911.000 | -3884.887 | 64.0 |
| `cubic-pm1` | 8,192 | `cpu-sa` | 7.015 | -3922.200 | -3897.125 | 64.0 |
| `cubic-pm1` | 8,192 | `cpu-msa-unit` | 0.806 | -3916.600 | -3897.062 | 64.0 |
| `cubic-pm1` | 8,192 | `cpu-msa-f64` | 8.888 | -3919.800 | -3897.375 | 64.0 |

### Time to matched quality

| Regime | Kernel | Scalar reference mean best at 8,192 | Reaching sweeps | Least median seconds | Mean best at reaching sweeps |
| --- | --- | ---: | ---: | ---: | ---: |
| `native-pm1` | `cpu-sa` | -13082.000 | 8,192 | 14.586 | -13082.000 |
| `native-pm1` | `cpu-msa-unit` | -13082.000 | not reached | not reached | not reached |
| `native-pm1` | `cpu-msa-f64` | -13082.000 | not reached | not reached | not reached |
| `fractional` | `cpu-sa` | -7958.445 | 8,192 | 14.727 | -7958.445 |
| `fractional` | `cpu-msa-f64` | -7958.445 | not reached | not reached | not reached |
| `multilevel` | `cpu-sa` | -59343.600 | 8,192 | 16.471 | -59343.600 |
| `multilevel` | `cpu-msa-f64` | -59343.600 | not reached | not reached | not reached |
| `clique-portfolio` | `cpu-sa` | -477.850 | 512 | 0.037 | -477.850 |
| `clique-portfolio` | `cpu-msa-f64` | -477.850 | 512 | 0.069 | -477.850 |
| `cubic-pm1` | `cpu-sa` | -3922.200 | 8,192 | 7.015 | -3922.200 |
| `cubic-pm1` | `cpu-msa-unit` | -3922.200 | not reached | not reached | not reached |
| `cubic-pm1` | `cpu-msa-f64` | -3922.200 | not reached | not reached | not reached |

### Quality at matched time

| Regime | Float anchor sweeps | Kernel | Median actual sweeps | Median seconds | Mean best energy | Mean energy | Mean distinct |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `native-pm1` | 512 | `cpu-msa-f64` | 512 | 1.287 | -13010.800 | -12951.550 | 64.0 |
| `native-pm1` | 512 | `cpu-sa` | 707 | 1.281 | -13031.600 | -12969.812 | 64.0 |
| `native-pm1` | 2,048 | `cpu-msa-f64` | 2,048 | 5.156 | -13058.000 | -13009.400 | 64.0 |
| `native-pm1` | 2,048 | `cpu-sa` | 2,883 | 5.149 | -13063.600 | -13020.325 | 64.0 |
| `native-pm1` | 8,192 | `cpu-msa-f64` | 8,192 | 20.650 | -13077.200 | -13043.475 | 64.0 |
| `native-pm1` | 8,192 | `cpu-sa` | 11,599 | 20.740 | -13091.600 | -13051.500 | 64.0 |
| `fractional` | 512 | `cpu-msa-f64` | 512 | 1.230 | -7891.729 | -7848.353 | 64.0 |
| `fractional` | 512 | `cpu-sa` | 653 | 1.220 | -7905.310 | -7858.795 | 64.0 |
| `fractional` | 2,048 | `cpu-msa-f64` | 2,048 | 4.896 | -7937.288 | -7903.334 | 64.0 |
| `fractional` | 2,048 | `cpu-sa` | 2,692 | 4.866 | -7936.501 | -7909.688 | 64.0 |
| `fractional` | 8,192 | `cpu-msa-f64` | 8,192 | 19.590 | -7956.842 | -7933.884 | 64.0 |
| `fractional` | 8,192 | `cpu-sa` | 10,928 | 19.589 | -7958.320 | -7938.512 | 64.0 |
| `multilevel` | 512 | `cpu-msa-f64` | 512 | 1.275 | -58752.000 | -58449.188 | 64.0 |
| `multilevel` | 512 | `cpu-sa` | 623 | 1.278 | -58860.400 | -58517.588 | 64.0 |
| `multilevel` | 2,048 | `cpu-msa-f64` | 2,048 | 5.086 | -59125.200 | -58853.925 | 64.0 |
| `multilevel` | 2,048 | `cpu-sa` | 2,526 | 5.072 | -59177.200 | -58908.062 | 64.0 |
| `multilevel` | 8,192 | `cpu-msa-f64` | 8,192 | 20.229 | -59322.800 | -59098.350 | 64.0 |
| `multilevel` | 8,192 | `cpu-sa` | 10,049 | 20.218 | -59357.200 | -59136.306 | 64.0 |
| `clique-portfolio` | 512 | `cpu-msa-f64` | 512 | 0.069 | -477.850 | -468.809 | 26.0 |
| `clique-portfolio` | 512 | `cpu-sa` | 979 | 0.068 | -477.850 | -469.727 | 22.2 |
| `clique-portfolio` | 2,048 | `cpu-msa-f64` | 2,048 | 0.277 | -477.850 | -471.414 | 16.8 |
| `clique-portfolio` | 2,048 | `cpu-sa` | 4,058 | 0.272 | -477.850 | -472.617 | 15.8 |
| `clique-portfolio` | 8,192 | `cpu-msa-f64` | 8,192 | 1.108 | -477.850 | -473.976 | 11.0 |
| `clique-portfolio` | 8,192 | `cpu-sa` | 16,315 | 1.098 | -477.850 | -474.468 | 10.2 |
| `cubic-pm1` | 512 | `cpu-msa-f64` | 512 | 0.556 | -3895.000 | -3864.438 | 64.0 |
| `cubic-pm1` | 512 | `cpu-sa` | 618 | 0.550 | -3899.800 | -3867.575 | 64.0 |
| `cubic-pm1` | 2,048 | `cpu-msa-f64` | 2,048 | 2.222 | -3911.000 | -3884.887 | 64.0 |
| `cubic-pm1` | 2,048 | `cpu-sa` | 2,563 | 2.238 | -3913.400 | -3886.588 | 64.0 |
| `cubic-pm1` | 8,192 | `cpu-msa-f64` | 8,192 | 8.888 | -3919.800 | -3897.375 | 64.0 |
| `cubic-pm1` | 8,192 | `cpu-sa` | 10,393 | 8.832 | -3917.400 | -3898.613 | 64.0 |

### Float kernel workspace

| Regime | Workspace bytes |
| --- | ---: |
| `native-pm1` | 1,360,932 |
| `fractional` | 1,360,932 |
| `multilevel` | 1,360,932 |
| `clique-portfolio` | 149,668 |
| `cubic-pm1` | 361,368 |

`workspace_bytes` is the float kernel's estimate of persistent workspace. It excludes total process memory and peak resident memory. The values were constant across the five models and three sweep counts within each regime. The other kernels report no workspace estimate.

## Findings

At matched sweeps, the float kernel took longer than scalar SA at every reported regime and sweep count. At 8,192 sweeps on `multilevel`, the float kernel took 20.229 seconds, while scalar SA took 16.471 seconds. On `clique-portfolio`, they took 1.108 and 0.552 seconds.

Compared with scalar SA's mean best energy at 8,192 sweeps, the float kernel reached the target only on `clique-portfolio`. Both kernels reached that target at 512 sweeps. Scalar SA took 0.037 seconds, and the float kernel took 0.069 seconds. The unit kernel did not reach the target on either regime where it ran.

At matched time, the float kernel had lower mean best energy at two of the 15 regime and sweep comparisons. Scalar SA had lower mean best energy at ten, and three were equal. The float wins were `fractional` at 2,048 anchor sweeps and `cubic-pm1` at 8,192 anchor sweeps. These comparisons average five model seeds and do not establish statistical significance.

The unit kernel was faster than scalar SA at each matched-sweeps point where both ran. On `cubic-pm1` at 2,048 sweeps, both had mean best energy of `-3908.600`. The unit kernel took 0.202 seconds versus scalar SA's 1.805 seconds. Equal quality at lower time is a win at this one-call timing boundary.

## Scope

The benchmark did not run on a quantum processing unit. Device-access time and end-to-end time for such hardware are outside its scope. The measured times cover one `sample_research` call and exclude model construction.
