# Parameter tuning

This directory documents how the controller's parameters were chosen. Nothing here
is needed to reproduce the paper's figures; it exists so the numbers in the configs
can be traced to the experiment that produced them.

| parameter | variants it applies to | script |
|---|---|---|
| `cbf_alpha` (CBF decay rate) | 02, 04, 05 | `sweep_gamma_lookahead.py` |
| `hocbf_gamma1`, `hocbf_gamma2` | 06 | `sweep_gamma_lookahead.py` |
| `w_soft` (slack penalty) | 03, 04, 05, 06 | `sweep_gamma_lookahead.py` |
| `lookahead_dist` (handle length `L`) | 05 only | `sweep_gamma_lookahead.py` |
| `d_safe`, movement headroom | all six | `occupancy_d_safe_sweep.py` + `analyze_occupancy_sweep.py` |
| `comm_radius`, neighbour cap `K` | all six | `comm_radius_sweep.py` + `analyze_comm_sweep.py` |
| `d_la` (navigation look-ahead) | all six | `lookahead_sweep.py` |

Variant 01 (hard barrier) has no tunable gain. It exposes only `d_safe`, which is the
safety specification rather than a knob, so it appears in the operating-point sweeps
but not in the gain sweep.

`sweep_gamma_lookahead.py` runs a screening grid, then Optuna TPE warm-started from
that grid, then a verification of the winner at the full protocol. The gains are
compiled into the solver, so each trial regenerates it and trials for one variant are
sequential; run different variants as separate jobs.

Caveat on `lookahead_sweep.py`: it evaluated `d_la` over 6 s runs rather than the full
horizon, and treated the existing 7 cm as the reference rather than deriving it. Of the
parameters listed above it has the weakest experimental support.
