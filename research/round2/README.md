# Round 2 research pipeline in quip-miner-cpu

This directory and quip-miner-dwave both hold a Round 2 pipeline. Each copy has one role.

| Copy | Role |
|---|---|
| quip-miner-dwave, `quip_miner_dwave/round2_runner.py` and `scripts/round2_report.py` | Canonical for the Round 2 evidence and the published Round 2 report. The frozen campaign ran its runner at sha256 `0ce48bf537f1…`, commit `1f9e452`. Its report adds the portfolio families section. |
| quip-miner-cpu, this directory | Home of the work that followed the campaign, from commit `575f5d4` through `9b54939`. That work adds a captured-model `dwave-neal` comparison and a matched-time QPU report. |

The copies have diverged, so do not copy one file over the other. Port a change by hand, and name the source commit.

This copy still reports the retired `clique-portfolio` regime. The quip-miner-dwave report replaces it with the portfolio families.
