# Simulated Bifurcation kernels

This document names the paper behind each Simulated Bifurcation (SB) kernel,
the constants the kernel uses, and every place the code departs from its
paper. All nine kernels share one integrator in `src/sb_core.rs`, one adapt
envelope, one seed contract, and the streaming pump. `num_sweeps` is the
step count. `sweeps_per_beta` and `beta_range` are ignored.

## Sign convention

The CSR stores quip's `j` and `h` unnegated. The force line
`y += (restore * x - c0 * f) * dt` carries the SB negation in its minus
sign, so a paper's `+ c sum_j J_ij g(x_j)` is this code's `- c0 * f`. Local
fields enter through one ancilla particle. The readout is `sign(x)` at the
last step, gauge-fixed by the ancilla sign.

## Kernels

| Binary | Method | Source |
|---|---|---|
| `quip-cpu-sb` | discrete SB (dSB) | Goto et al., Sci. Adv. 7, eabe7953 (2021) |
| `quip-cpu-bsb` | ballistic SB (bSB) | same |
| `quip-cpu-hdsb`, `quip-cpu-hbsb` | heated dSB, heated bSB | Kanao and Goto, Commun. Phys. 5, 153 (2022) |
| `quip-cpu-gbsb` | edge-of-chaos control on bSB (GbSB) | Goto, Hidaka, Tatsumura, Phys. Rev. Applied 25, 044011 (2026), arXiv:2508.17655 |
| `quip-cpu-gdsb` | the same control on dSB | this project's extension. The paper names it as future work |
| `quip-cpu-tedsb` | tabu-enhanced dSB (TEdSB) | Tao et al., Commun. Phys. 9, 100 (2026). Code: github.com/Tao-qubit/Tabu-Enhanced-Simulated-Bifurcation |
| `quip-cpu-sbqa` | replica-ring dSB (SBQA) | Pawlowski et al., arXiv:2604.01050. Data: github.com/quantumz-io/SBQA_benchmarks |
| `quip-cpu-ggdsb` | globally guided dSB (GGdSB) | Xiao et al., Phys. Rev. Applied 25, 024014 (2026). Code: github.com/JugarH/Global-Guided-Simulated-Bifurcation |

Every binary in the table except `quip-cpu-sb` builds behind
`--features experimental`. A Release publishes all of them and names the
experimental ones in its notes.

## Edge-of-chaos control: `quip-cpu-gbsb`, `quip-cpu-gdsb`

Each particle carries its own bifurcation parameter `p_i`, started at 1 and
advanced once per step from the pre-step position:

```text
p_i -= (1 - A x_i^2) p_i / (M - k)
y_i += (-p_i x_i - c0 f_i) dt
```

A particle at a wall keeps its pump longer. With `A = 0` the parameter
equals the scalar pump `1 - (k+1)/M` and the plain variants are recovered.
The final `p_i` is `A x_i^2 p_i(M-1)`, a residual restoring force that the
paper's mechanism depends on.

Constant: `EDGE_OF_CHAOS_CONTROL = 0.2`. The paper tunes `A` per instance
from 0.04 to 0.66 and uses 0.2 in its step-size study. The campaign re-tunes.

Three departures. The initial momentum stays `U(-0.1, 0.1)` where the paper
uses zero. The coupling scale stays `c0` where the paper uses
`1 / lambda_max(J)`. The ancilla is an ordinary particle with its own `p`.

Every instance the paper reports has zero fields, at most 2000 nodes, and a
density of at least 6 percent. On G9 it loses to plain dSB.

## Tabu-enhanced dSB: `quip-cpu-tedsb`

Two phases in one job. A warm-up runs 100 plain dSB replicas over the first
10 percent of the steps and stores each final configuration as a column of a
tabu list. The checking phase restarts `num_reads` replicas over the
remaining 90 percent. At each step two stored columns, drawn with
replacement, are averaged into a field `tau`, and every replica feels
`- c0 beta tau_i`, a push away from configurations already found. The draw
is shared across replicas at a step, as in the paper's Algorithm 1.

Constants (`TesbConfig`): `alpha = 0.9`, `beta = 1`, `mini_batch = 2`,
`warm_replicas = 100`, `check_c0_scale = 1`. The paper picks the checking
`c0` per instance from `{0.02, 0.05, 0.08, 0.1}`. Its degree-20 G-set
instances have the same theoretical `c0` as a 4577-node Zephyr instance,
so that grid transfers.

Three departures. The paper's Eq. 10 shifts the restoring coefficient by
`c0 beta`. Both released implementations omit that shift and every published
number comes from them, so this kernel omits it too. Stored columns are
gauge-fixed by the ancilla sign and the ancilla is dropped. Each step
multiplies the field by the replica's own ancilla sign, and the ancilla
receives no tabu force. With zero fields this is the published algorithm
exactly. The initial draw stays `U(-0.1, 0.1)` where the reference uses
`U(-0.005, 0.005)`.

With `beta = 0` the checking phase is plain dSB bit for bit at the same
per-read seeds. The paper's gains on degree-20 G-set instances come from the
discrete form. Its ballistic form fails where plain bSB fails, so this
project does not ship it.

## Replica-ring dSB: `quip-cpu-sbqa`

Reads form consecutive periodic rings of `replicas` trajectories. Each
replica runs dSB with a dead zone, `f(x) = 0` for `|x| <= 0.7 t/T` and
`sign(x)` outside it, and at every step adds a ferromagnetic force from its
two ring neighbours:

```text
theta = theta0 ((1 - u)^alpha + 1e-5)
kring = R (-(1 / 2 beta)) ln tanh(theta)
y_{i,q} += (restore x_{i,q} - c0 F_{i,q} + kring (x_{i,q-1} + x_{i,q+1})) dt
```

`kring` is positive and rises through the run, so replicas explore
independently early and the ring tightens late. `beta ~ U(0.5, 1.5)` and
`alpha ~ U(0.5, 1.0)` are drawn once per ring.

Constants (`SbqaConfig`): `replicas = 16`, `theta0 = 2.5`, `dead_zone = 0.7`.
The paper states its replica count (128) only for a sensitivity study and
never states `Gamma_x(0)`. Only `theta = beta Gamma_x / R` enters the
schedule, so `theta0` is the exposed parameter. The paper's companion data
fix the step size: the SB part advances at plain dSB's rate per step, which
absorbs the paper's `1/R` factors into `kring = R J_perp`.

Rings of one replica have no ring term. A ring does not always reach
consensus: two adjacent replicas that disagree with the rest feel no net ring
force, and a ring of two swaps sides every step once `kring` exceeds the
symplectic stability limit at `dt = 1`. Read diversity inside a ring is low,
not zero. The paper's Zephyr benchmark uses uniform `[-1, 1]` fields and
couplings and shows the method overtaking plain SB above about 4000 nodes.
Its closest match to a mining instance, 3D spin glasses embedded in Pegasus
at 5400 nodes, shows a 128-fold time-to-target gain at a tight target.

## Globally guided dSB: `quip-cpu-ggdsb`

All `num_reads` trajectories advance in lockstep. Each step, every
trajectory's extended energy is read off the coupling sum it already
computed, the lowest becomes the leader, and every trajectory is pulled
toward the leader's continuous position:

```text
guid = (w / (1 + ||gbest - x||)) (gbest - x), negated as a whole if cos(guid, y) < -b
vm   = alpha vm + (1 - alpha) guid
y   += (restore x - c0 f + vm) dt;  x += y dt
with probability q per coordinate: x -= a;  then the wall
q = 0.1 + 0.45 (1 + cos(pi u)),  alpha = 0.9 - 0.4 u,  w = 1 - u
```

Constants (`GgsbConfig`): `perturb_amplitude = 0.01`, `flip_threshold = 0.05`,
the reference demo's values for its discrete form. The schedules are the
reference's literals.

The paper text is behind a paywall with no preprint. The kernel is
transcribed from the reference code the paper cites as its own.

Four departures. The linear pump of this crate replaces the reference's
default exponential pump, which the reference also supports. `c0` replaces
the reference's coupling scale, whose automatic path is dead code. With a
bias, the leader is aligned to each trajectory's ancilla sign before the
difference is taken. The leader is the current-step best, which is what the
reference computes after its first improving step.

With one read and no perturbation the kernel is plain dSB bit for bit.

The reference's guidance gain uses an unnormalized norm over all
coordinates, so the pull weakens as about `1 / sqrt(N)` between the 800 to
2000-node instances of its tuning and a 4577-node instance. If the kernel
underperforms, a per-coordinate RMS in the gain is the first knob to add.

## What the literature says about sparse graphs

Hou, Barzegar and Katzgraber (Phys. Rev. E 112, 035301, 2025) measure SB
time-to-solution rising as graph connectivity falls, with connectivity 16
the worst point tested. Zephyr has mean degree 18. Zeng et al. (Commun. Phys.
7, 249, 2024) find dSB reaching the ground state on full Pegasus graphs with
zero fields where D-Wave hardware does not. No paper benchmarks any SB variant on
Zephyr with ternary couplings. The numbers above are starting points for the
campaign, not results.

## Not implemented, and why

- AE-QSB (arXiv:2607.02540): seven constants its equations need are never
  stated, three of its periodic triggers cannot fire as written, and its
  median gap on G-set is worse than plain dSB.
- TAC (Nat. Commun. 14, 2510, 2023): a readout post-process rather than a
  dynamics change, with dSB evidence limited to a qualitative supplementary
  figure on dense graphs.
- hSB (arXiv:2404.08265): its force reduces to dSB's on any quadratic model.
