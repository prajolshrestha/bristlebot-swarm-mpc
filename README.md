# Distributed Predictive Flocking for a Swarm of Miniature Vibration-Driven Robots in a Confined Arena

## Overview

Coordinating many miniature robots in a confined arena, with no leader, central planner
or global map, is a core challenge for swarm robotics. As the swarm grows denser,
collision avoidance becomes the limiting factor, and how many robots can safely coexist
depends on how the inter-robot safety constraint is formulated.

Each robot solves its own nonlinear model predictive control problem, combining
Reynolds-style flocking, inter-robot and obstacle safety, and a task objective. Within
that identical controller, this repository implements **six barrier and
control-barrier-function formulations** that differ *only* in the safety constraint, so
comparing them isolates that one design choice. The plant is a digital twin of an
open-source vibration-driven bristlebot in a 0.8 m x 0.8 m arena.

## The six safety formulations

| directory | label | safety constraint |
|---|---|---|
| `01_hard_bf` | Hard BF | barrier as a hard constraint |
| `02_hard_dt_cbf` | Hard Dt-CBF | discrete-time CBF, hard |
| `03_slacked_bf` | Slack BF | barrier with slack |
| `04_slacked_dt_cbf` | Slack Dt-CBF | slacked barrier anchor + slacked Dt-CBF rate | 
| `05_slacked_dt_cbf_with_lookahead` | LA Slack Dt-CBF | the same, rate evaluated on a forward handle | 
| `06_slacked_dt_hocbf` | Slack Dt-HOCBF | the same, with a two-step higher-order chain |

The last three share one structure and differ only in the rate condition. That isolates
the two responses to the relative-degree obstruction a distance constraint has under
unicycle dynamics: move the constraint onto a forward handle, or extend it into a
higher-order chain.

The last column is the internal name the harness uses. It appears in the `--variants`
argument and in the `safety_level` column of every results CSV, so it is what you will
see when reproducing the figures.

## Collective behaviors

The safety formulation and the task are independent, so any variant can drive any
behavior. A behavior only changes the reference each robot tracks: the behavior module
turns a robot's local light reading into a look-ahead reference pose, and the same
optimal control problem then tracks that pose while enforcing flocking and safety. No
robot knows where the light source is, and none communicates a goal.

| `--behavior` value | what the swarm does |
|---|---|
| `phototaxis` | climb the sensed intensity gradient toward the source |
| `phototaxis_no_reynolds` | phototaxis with the flocking terms removed |
| `phototaxis_orbital` | follow the tangent of the intensity level set, orbiting the source |
| `no_light` | field off, so there is no task; the flocking terms alone set the motion |
| `phototaxis_orbital_contracting` | orbit while contracting toward the source; used by the obstacle matrix |


The first two are the tasks. The last two are controls that isolate the two halves of
the controller: `no_light` removes the task and keeps flocking, `phototaxis_no_reynolds`
removes flocking and keeps the task. The latter collapses the swarm, because every robot
converges on the same peak and piles up regardless of swarm size, which is what motivates
the coupling terms in the first place.

Which campaign uses which:

- the safety comparison (Figs. 3 and 4, Table 2) runs `phototaxis` only, so the six
  formulations are compared on a single task;
- the obstacle matrix (Figs. 6 and 7) runs phototaxis, orbiting, the contracting
  orbit and the light-off control across three environments;
- the snapshot figures (Figs. 5, A.1, A.2) run `phototaxis`,
  `phototaxis_no_reynolds`, `phototaxis_orbital` and `no_light` side by side.

`sim_engine/swarm_dynamics.py` holds the full enum, including a fixed-centre orbit mode
which is outside the scope of this paper. Note the campaign drivers and the demo
recorder use different spellings for the light-off control: the behavior enum and the
campaign jobs call it `no_light`, while `make_demo_videos.py` accepts it as
`no_stimulus`, matching the wording used in the paper.

## Requirements

Linux or WSL with `git`, `cmake`, `make`, `gcc`/`g++` and `conda`. The optimal control
problems are compiled with [acados](https://github.com/acados/acados), which `setup.sh`
builds for you. Python dependencies are in `requirements.txt`.

## Installation

```bash
git clone https://github.com/prajolshrestha/bristlebot-swarm-mpc.git
cd bristlebot-swarm-mpc
./setup.sh --compile-solvers
```

This builds acados, creates the `acados_casadi` conda environment, installs the Python
dependencies, writes `env.sh`, and compiles the six solvers. Then, in every shell:

```bash
source env.sh
conda activate acados_casadi
```

Variations:

```bash
./setup.sh --skip-acados --compile-solvers   # acados already installed
./setup.sh --verify                          # check tools, environment and solvers
```

Compiled solvers are not distributed. acados bakes an absolute build path into each one,
so they are not portable between machines and must be generated locally.

## Verifying the installation

```bash
cd src/experiments/simulation
sbatch jobs/cluster/smoke_test.sh
```

Loads every variant's solver in a separate process, then runs the campaign, aggregate,
figure and table chain at a short horizon. A few minutes, and worth running after any
move, rename or environment change.

## Reproducing the results

Everything is driven from `src/experiments/simulation/`, whose
[README](src/experiments/simulation/README.md) maps each figure and table to the script and command
that produce it. In outline:

```bash
cd src/experiments/simulation
sbatch --job-name=bench  jobs/cluster/run_bench.sh phototaxis
sbatch --job-name=dist   jobs/cluster/run_distance.sh
sbatch --job-name=matrix jobs/cluster/run_matrix.sh 05
sbatch                   jobs/cluster/run_snapshots.sh
# once all four have drained:
sbatch jobs/cluster/run_figures.sh        # every figure, into results/figures/
sbatch jobs/cluster/make_paper_artifacts.sh   # the two tables
```

That is 1692 simulations of 900 s each. The three sweep campaigns run 10 random seeds
at swarm sizes N = 10, 25, 50, 75, 100, 125, 137; the snapshot campaign is 12 runs at
N = 50 on a single seed, since it produces figures rather than statistics. Run
concurrently on a cluster it takes about four hours. `jobs/cluster/run_figures.sh` then renders the nine data-derived figures into
`results/figures/`, in about three minutes; it reads the aggregated data and runs no
simulation. `make_paper_artifacts.sh` produces the two tables. The tenth figure in the
paper, the architecture diagram, is drawn by hand and has no generator here.

Submit every job from `src/experiments/simulation`, as above. Slurm resolves each
script's relative `--output` path against the directory you submit from and creates it
if missing, so submitting from elsewhere quietly scatters a stray `results/logs/` there.
Each job checks its submission directory up front and exits rather than let that happen.

Output goes to `results/csv/` (aggregated data), `results/figures/` and
`results/tables/`, with per-run traces under `results/raw/` and one Slurm log per task
under `results/logs/`. None of it is tracked by git, since all of it regenerates from
the commands above; only an empty `results/logs/.gitkeep` is committed, because Slurm
will not create a missing `--output` directory and a job whose log cannot be opened
dies at launch.

### Running without a cluster

Installation is identical to the cluster path; verify it with the
local smoke test rather than the sbatch one:

```bash
./jobs/local/smoke_test_local.sh   # a couple of minutes
```

It loads every variant's solver in its own process, runs one short simulation through the
campaign driver, generates both tables, and imports every figure script. It skips the
bench aggregate on purpose: that step needs the whole 420-run grid and would quietly
simulate whatever is missing.

`sbatch` does not exist on a workstation, so the four campaign jobs cannot run there.
Everything under them can: the wrappers, figure generators and table scripts are plain
Python that takes explicit arguments. `src/experiments/simulation/jobs/local/` replaces each Slurm
array with a shell loop parallelised across local cores.

```bash
cd src/experiments/simulation/jobs/local
./run_all.sh                  # all four campaigns, then the tables and figures
```

or one campaign at a time with `run_bench.sh`, `run_distance.sh`, `run_matrix.sh`,
`run_snapshots.sh`. 

**These default to a reduced protocol**, because the published one is not feasible on a
workstation: 1692 runs of 900 s is roughly 1400 CPU-hours, about a week on eight cores.
The defaults are 90 s runs, 3 seeds and 3 densities, which is on the order of an hour.

| knob | local default | paper |
|---|---|---|
| `STEPS` | 900 (90 s) | 9000 (900 s) |
| `SEEDS` | 3 | 10 |
| `NS` | `10 50 137` | `10 25 50 75 100 125 137` |
| `JOBS` | all cores | one task per core |

Override any of them to move toward the published protocol:

```bash
STEPS=9000 SEEDS=10 NS="10 25 50 75 100 125 137" ./run_all.sh
```

A reduced run reproduces the *shape* of every figure: the density trends, the ranking of
the formulations at high density, and the collapse of local order in the light-off
control. It does **not** reproduce the published numbers, and the confidence bands are
much wider at three seeds.

The snapshot figures are the exception: 12 simulations, affordable at the full 900 s
protocol even on a laptop, so `run_snapshots.sh` does not reduce anything.

### Visualization of different behaviors (demo) 

A workstation can do something the cluster cannot: show the swarm moving. Each variant
ships a live viewer that runs the same simulation as the campaigns, with rendering on
top.

```bash
cd src/deterministic_bristlebot_swarm/05_slacked_dt_cbf_with_lookahead
python sim_engine/viz_engine.py
```

The behavior and swarm size are module constants near the top of `viz_engine.py`
(`BEHAVIOR_MODE`, `N_ROBOTS`), so pick one of `PHOTOTAXIS`, `PHOTOTAXIS_ORBITAL`,
`PHOTOTAXIS_ORBITAL_CONTRACTING`, `NO_LIGHT` or `PHOTOTAXIS_NO_REYNOLDS` and rerun.
Watching `PHOTOTAXIS_NO_REYNOLDS` next to `PHOTOTAXIS` is the quickest way to see why
the flocking terms are there: without them every robot drives at the same peak and
piles up.

This needs a display. On WSL2 that means WSLg or an X server, and `MPLBACKEND` must not
be set to `Agg`.

To record instead of watch, which works headless:

```bash
python sim_engine/make_demo_videos.py --behavior phototaxis --behavior orbital \
       --env free --duration 60 --n 50
```

`--behavior` accepts `phototaxis`, `orbital`, `poc`, `no_stimulus` and `no_reynolds`, and
may be repeated. Any variant works, not just the deployed one; running the same behavior
under two safety formulations shows what the constraint changes.

There is also a doorway scenario, which is the most legible
demonstration of the safety constraint: robots spawn one at a time and have to file
through a gap narrow enough that only one fits, so the avoidance behavior is visible
rather than inferred from a collision count.

```bash
cd src/deterministic_bristlebot_swarm/05_slacked_dt_cbf_with_lookahead/sim_engine
python spawn_sim.py --behavior phototaxis --env free --door corner_low --duration 900
python spawn_render.py --behavior phototaxis --env free
```

The simulation writes trajectories to `door_runs/`, and the renderer turns them into a
video. `--door` picks the gap (`bottom`, `corner`, `corner_low`), `--nmax` the number of
robots and `--spawn-interval` how fast they arrive. `--force-spawn` admits a robot on
every interval even while the doorway is occupied, instead of waiting for it to clear.
Pass `--env` to the renderer to draw one arena rather than all three side by side. `spawn_render_hero.py` renders
the same run as a single showcase frame instead of a video.

### Supplementary videos

The videos accompanying the paper are built in two steps from
`src/experiments/simulation`. The first renders one clip per behavior and arena; the
second stacks each arena's four clips into a montage with a title card, a shared legend
and a still.

```bash
sbatch jobs/cluster/run_videos.sh                  # 12 clips, 900 s each
python behaviors_src/make_video_montages.py        # 3 montages + stills + captions
```

Clips land in `results/videos/`, montages in `results/videos/montage/` alongside a
`SUPPLEMENTARY.md` holding each file's caption and measured specification. The montage
script takes `--show-seconds` to window the clips without re-rendering, and `--out-width`
and `--crf` to control the encode.

## Repository layout

```
src/
  deterministic_bristlebot_swarm/   the six controller variants
    NN_.../expert_src/              OCP model, acados solver generator, controller
    NN_.../sim_engine/              headless and interactive simulators
    NN_.../test/                    unit and behavior tests
  experiments/simulation/
    comparison_src/                 cross-variant drivers, figure and table generators
    behaviors_src/                  obstacle-matrix, coordination and snapshot drivers
    jobs/cluster/                   sbatch scripts, smoke test, artifact builder
    jobs/local/                     the same campaigns without Slurm
    tuning/                         how the gains and the operating point were chosen
    results/                        csv, figures, tables, raw (all regenerable)
setup.sh                            one-command install
env.sh                              acados environment, written by setup.sh
```

Each variant is self-contained and differs from the others only in its safety
formulation. The flocking rules, task objectives, simulator and metrics are shared.

## Notes for users

- Runtime parameters (communication radius, consensus iterations, cost weights) are read
  live from each variant's `expert_src/config/solver_ellipse_mpc.yaml`. The safety gains
  (`cbf_alpha`, `lookahead_dist`, `hocbf_gamma1/2`) are compiled into the solver and need
  it regenerated after a change.
- Every variant ships a Python package named `expert_src`, so two variants cannot be
  imported into one interpreter. Each simulation runs in its own process.
- `src/experiments/simulation/tuning/` records how each parameter was selected, and is explicit
  about which selections rest on weaker evidence than others.

## Citation

A BibTeX entry will be added once the paper has a DOI.

## License

MIT, see [LICENSE](LICENSE). The dependencies keep their own licences: acados and
CasADi are LGPL-2.1 and LGPL-3.0 respectively, so a redistributed binary that links
them carries those terms.

## Acknowledgements

Optimal control problems are compiled with [acados](https://github.com/acados/acados)
and [CasADi](https://web.casadi.org/); the hyperparameter searches use
[Optuna](https://optuna.org/).
