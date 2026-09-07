# Experiments

Everything needed to reproduce the results in the paper. Each figure and table below
names the script that produced it and the command that runs it.

All simulations are Slurm batch jobs. Analysis and plotting are light enough for a login
node. Activate the environment first:

```bash
source env.sh              # from the repository root
conda activate acados_casadi
```

The solvers are not in the repository (acados bakes an absolute build path into each
one, so a copied solver is not portable). Build them once:

```bash
./setup.sh --compile-solvers
```

## Layout

| directory | contents |
|---|---|
| `comparison_src/` | cross-variant drivers, the benchmark harness, and the figure and table generators |
| `jobs/cluster/` | sbatch scripts: one per campaign, plus the smoke test and the artifact builder |
| `jobs/local/` | the same campaigns as shell loops, for a workstation without Slurm |
| `behaviors_src/` | the deployed variant's behavior, obstacle-matrix and snapshot drivers |
| `tuning/` | how the safety gains and the operating point were chosen (see its own README) |
| `results/csv/` | aggregated CSVs, the record behind every figure |
| `results/figures/` | rendered figures |
| `results/raw/` | per-run traces and checkpoints |

Scripts inside one directory import each other laterally, so keep each group together.
Everything under `results/` is regenerable and is not tracked by git.

## The six formulations

| key | directory | paper label |
|---|---|---|
| 01 | `01_hard_bf` | Hard BF |
| 02 | `02_hard_dt_cbf` | Hard Dt-CBF |
| 03 | `03_slacked_bf` | Slack BF |
| 04 | `04_slacked_dt_cbf` | Slack Dt-CBF |
| 05 | `05_slacked_dt_cbf_with_lookahead` | LA Slack Dt-CBF, the deployed one |
| 06 | `06_slacked_dt_hocbf` | Slack Dt-HOCBF |

Every campaign runs 900 s (9000 steps at dt = 0.1 s) over 10 seeds, at
`N_b` = 10, 25, 50, 75, 100, 125, 137.

## Reproducing the paper

### Fig. 3 and Table 2 - safety formulation comparison

All six formulations under phototaxis in free space: feasibility, solve time, target
distance and collisions against swarm size. 420 runs.

```bash
sbatch --job-name=bench jobs/cluster/run_bench.sh phototaxis
# then, once the array has drained:
cd comparison_src
python bench_wrapper.py --aggregate --behavior phototaxis \
       --variants "Hard BF,Hard Dt-CBF,Slack BF,Slack Dt-CBF,LA Slack Dt-CBF,Slack Dt-HOCBF" \
       --results-dir ../results/raw/bench_900s
python paper_comparison_figure.py --case phototaxis --res-dir ../results/raw/bench_900s
python make_paper_table.py            # Table 2, LaTeX rows to stdout
```

`bench_wrapper.py` defaults to the 900 s protocol; the
harness itself is `auto_benchmarking_parallel.py`. The behavior argument defaults to
`phototaxis`, which is the campaign the paper reports.

### Fig. 4 - safety margins

Same six formulations and grid, obstacles on, recording the closest robot-robot and
robot-obstacle approach ever reached. 420 runs.

```bash
sbatch --job-name=dist jobs/cluster/run_distance.sh
cd comparison_src
DIST_RESULTS_DIR=$PWD/../results/raw/distance_900s python distance_wrapper.py --merge
```

The merge writes `phototaxis_distances_vs_density.csv` and `safety_margins.{png,pdf}`
in one step, so there is no separate figure script. Per-run work is checkpointed as
pickle fragments, and a resubmitted array resumes.

### Figs. 5, 8 and 9 - behavior snapshots

Four behaviors in three environments (0, 2 and 12 obstacles) at `N_b` = 50, sampled at
t = 0, 225, 450, 675 and 900 s. 12 runs, one seed, since these illustrate rather than
measure.

```bash
sbatch jobs/cluster/run_snapshots.sh
```

which runs `behaviors_src/collect_behavior_obstacle_snapshots.py` and then
`behaviors_src/render_obstacle_grouped_figs.py`.

Both of the next two figures come from one campaign: the deployed variant across four
behaviors x three obstacle counts x seven densities x 10 seeds = 840 runs.

```bash
sbatch --job-name=matrix jobs/cluster/run_matrix.sh 05
```

The simulation driver is `behaviors_src/steady_state_matrix.py`, reached through
`comparison_src/matrix_wrapper.py`, which sets the output roots: CSVs to
`results/csv/matrix/`, raw per-robot state to `results/raw/matrix_npz/`.

### Fig. 6 - local order and nearest-neighbour distance

Local polar order and NND against swarm size, one column per behavior, one curve per
obstacle count. Reads the matrix CSVs, so run that campaign first.

```bash
cd comparison_src
python fig_order_nnd.py --both
```

`--both` adds `order_nnd_by_obstacle`, the transpose with columns by obstacle count. It
is not a paper figure; the paper uses `order_nnd_by_behavior`.

`matrix_figures.py --coordination` writes `coordination_order`, a different figure of
four order parameters that the current paper does not use.

### Fig. 7 - collisions across density and environment

Contacts against swarm size on a log axis, one panel per behavior, one curve per
obstacle count. Same campaign as Fig. 6.

```bash
cd comparison_src
python matrix_figures.py --collisions2 --summary --variant 05
```

`--collisions2` is the paper figure, `behavior_collisions_by_behavior`. Plain
`--collisions` writes `behavior_collisions`, an earlier three-environment version that
the current paper does not use.

### Occupancy table

```bash
python comparison_src/swarm_occupancy_table.py    # LaTeX to stdout, no simulation needed
```

### Solve-time claims

`timing_control.py` measures uncontended solve times, one solve at a time on an idle
node, so the numbers are not inflated by workers competing for cores.

```bash
sbatch jobs/cluster/run_timing.sh
```

## Checking the wiring

```bash
sbatch jobs/cluster/smoke_test.sh
```

Loads every variant's solver in a fresh subprocess, then runs the campaign, aggregate,
figure and table chain at a short horizon. Takes a few minutes and is worth running
after any move, rename or environment change.

## Notes

- Runtime parameters (communication radius, consensus iterations, cost weights) are read
  live from each variant's `expert_src/config/solver_ellipse_mpc.yaml`. The safety gains
  (`cbf_alpha`, `lookahead_dist`, `hocbf_gamma1/2`) are compiled in and need
  `generate_solver_via_acados.py` rerun after a change.
- acados bakes an absolute path into each generated solver, so a variant directory that
  is moved or renamed must have its solver regenerated, or its `code_export_directory`
  repointed, before anything will load.
- Every variant ships a package called `expert_src`, so two variants cannot be imported
  into the same interpreter. Each run gets its own process; the harness and the smoke
  test both fork per variant for this reason.
- Cap concurrency at 8 workers per node. Sharing cores between solves distorts timing.

## Supplementary videos

Two campaigns live under `results/videos/`. `random_initial_position/` is the one the
paper uses: 50 robots placed at random, four behaviors in three obstacle environments.
`spawn/` is a separate study in which robots enter one at a time through a door in the
bottom-left corner until the arena reaches its packing limit of 137.

### The paper set

```bash
sbatch jobs/cluster/run_videos.sh              # 12 clips, 900 s each
python behaviors_src/make_video_montages.py    # 3 montages + stills + captions
```

Clips land in `results/videos/random_initial_position/original/`, montages beside them in
`montage/`, together with a `SUPPLEMENTARY.md` holding each file's caption and measured
specification. Everything under `results/` is gitignored.

Defaults give 900 s of simulation at 5x real time: 2400x2700, 25 fps, CRF 34, 182 s per
montage. Useful options:

| flag | effect |
|---|---|
| `--show-seconds N` | window the clips to their first N simulated seconds, no re-render |
| `--out-width`, `--crf` | control the encode |
| `--clip-dir`, `--prefix`, `--n-robots` | build montages from a different campaign |
| `--entry` | extra title-card line describing how robots enter |

The submitted set is the 180 s window at higher quality:

```bash
python behaviors_src/make_video_montages.py --show-seconds 180 --fps 50 --crf 24 \
       --outdir results/videos/random_initial_position/montage_180s
```

Two smaller sets exist for contexts where 2400x2700 is too large: `make_row_montages.py`
lays the four behaviors in a single row and `make_individual_videos.py` writes one video
per behavior, both at 492x276 and roughly 750 kbps.

### The door-entry set

```bash
sbatch jobs/cluster/run_spawn_videos.sh        # 12 clips, 137 robots each
sbatch jobs/cluster/run_spawn_montages.sh      # 3 montages
```

The simulation is `spawn_sim.py` in the deployed variant's `sim_engine/`, rendered by
`spawn_render.py`. `--force-spawn` admits a robot on every interval even while the
doorway is occupied; without it the doorway must clear first, which is a different
experiment and yields different occupancy numbers.

Note the montage window matters here in a way it does not for the paper set: at one
robot per 5 s, the first 180 s reaches only 37 of 137 robots, so a 180 s montage shows
the arena filling rather than full.

### Rendering cost

A clip is rendered frame by frame, so the videos take far longer than the figures: about
2.5 to 4 hours per task for the door-entry set at 137 robots. The clips are the only
copy, since the recordings they are made from are deleted after rendering.
