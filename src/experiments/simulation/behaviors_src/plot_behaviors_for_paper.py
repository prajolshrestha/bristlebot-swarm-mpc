"""Draws the individual behavior panels used in the snapshot sheets."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import csv
import json
import yaml
from scipy.spatial import KDTree

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VARIANT_ROOT = os.path.abspath(os.path.join(
    _SCRIPT_DIR, "..", "..", "..", "deterministic_bristlebot_swarm",
    "05_slacked_dt_cbf_with_lookahead"))
_SIM_ENGINE_DIR = os.path.join(_VARIANT_ROOT, "sim_engine")
_COMPARISON_CODE_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "comparison_src"))
sys.path.insert(0, _VARIANT_ROOT)
sys.path.insert(0, _SIM_ENGINE_DIR)
sys.path.insert(0, _COMPARISON_CODE_DIR)

from expert_src.Ellipse2dObj import EllipticalBot2D
from expert_src.controller import CasadiAcadosEllipticalController, D_SAFE, D_SAFE_OBS
from sim_engine.swarm_dynamics import ReynoldsBehavior, SwarmBehavior
from sim_engine.light_source_model import LightField, SourceMotionMode
from sim_engine.obstacle_model import ObstacleManager, StaticObstacle, MovingObstacle
from sim_engine.collision_response import CollisionHandler, _ellipse_overlap_fast
from sim_engine.headless_sim_engine import HeadlessSim as _CanonicalHeadlessSim, MAX_CONSENSUS

_CMP_SRC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "comparison_src"))
if _CMP_SRC not in sys.path:
    sys.path.insert(0, _CMP_SRC)
from comparison_config import (
    FONTSIZE_AXIS,
    FONTSIZE_PANEL_TITLE,
    FONTSIZE_SUBFIGURE,
    FONTSIZE_SUPTITLE,
    FONTSIZE_TICK,
    PLOT_DPI,
    add_figure_legend,
    add_subfigure_label,
    configure_publication_style,
)

configure_publication_style()

_CFG_PATH = os.path.join(_VARIANT_ROOT, "expert_src", "config", "solver_ellipse_mpc.yaml")
with open(_CFG_PATH) as _cf:
    _CFG = yaml.safe_load(_cf)
_CFG_REYNOLDS = _CFG.get("reynolds", {})
_CFG_COLLISION = _CFG.get("collision", {})
_CFG_OBSTACLE = _CFG.get("obstacle_avoidance", {})
_CFG_SOLVER = _CFG.get("solver_creation", {})
_CFG_BOUNDS = _CFG.get("model_bounds", {})

DT = float(_CFG_SOLVER.get("Ts", 0.1))
MPC_HORIZON = int(_CFG_SOLVER.get("N", 20))
ARENA_HALF = float(_CFG_BOUNDS.get("px_max", 0.4))
WALL_OUTSET = 0.05
WALL_HALF = ARENA_HALF + WALL_OUTSET
ROBOT_A = 0.03
ROBOT_B = 0.015
MAX_STEPS = 600
SOURCE_MOTION_MODE = SourceMotionMode.ROTATION
SOURCE_ROT_RADIUS = 0.25
SOURCE_PULSING = False

GLOBAL_W_SOFT     = float(_CFG_COLLISION.get("w_soft", 1e8))
GLOBAL_Q_COH      = float(_CFG_REYNOLDS.get("q_cohesion", 2.0))
GLOBAL_Q_ALI      = float(_CFG_REYNOLDS.get("q_align", 5.0))
GLOBAL_W_SOFT_OBS = float(_CFG_OBSTACLE.get("w_soft_obs", 1e8))

OBSTACLE_RADIUS   = 0.03
ENABLE_OBSTACLES  = True

OBSTACLE_DEFS = [
    {"type": "static", "pos": [ 0.15,  0.10], "active": True},
    {"type": "static", "pos": [-0.15, -0.10], "active": True},
]

def _build_obstacle_manager(use_obstacles: bool | None = None) -> "ObstacleManager | None":
    """Build an ObstacleManager from OBSTACLE_DEFS. Returns None if disabled."""
    if use_obstacles is None:
        use_obstacles = ENABLE_OBSTACLES
    if not use_obstacles:
        return None
    mgr = ObstacleManager(dt=DT, arena_half=ARENA_HALF, max_obs=5, obs_range=0.30)
    for d in OBSTACLE_DEFS:
        if not d.get("active", True):
            continue
        pos = np.array(d["pos"], dtype=float)
        if d["type"] == "static":
            mgr.add(StaticObstacle(pos=pos, radius=OBSTACLE_RADIUS))
        elif d["type"] == "moving":
            vel    = np.array(d.get("vel", [0.0, 0.0]), dtype=float)
            bounce = bool(d.get("bounce", True))
            mgr.add(MovingObstacle(pos=pos, vel=vel, radius=OBSTACLE_RADIUS,
                                   arena_half=ARENA_HALF, bounce=bounce))
    return mgr

ENABLE_COLLISION = True
COLLISION_STEER_DECAY = 0.85
COLLISION_FLASH_FRAMES = 6
VIOL_COLOR = "#CC00CC"
ARENA_EDGE_COLOR = "black"
PANEL_BG_COLOR = "white"

SNAPSHOT_TIMES_SEC = (0, 15, 30, 45, 60)
SNAPSHOT_STEPS = tuple(int(round(t / DT)) for t in SNAPSHOT_TIMES_SEC)
PAPER_N_LIST = [25, 50, 75, 100]

ROBOT_COLORS = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
    "#56B4E9", "#F0E442", "#000000", "#999999", "#882255",
]
PUBLICATION_ROBOT_COLOR = "#0072B2"
HEADING_ARROW_LENGTH = 0.012
HEADING_ARROW_LW = 1.0
HEADING_ARROW_HEAD = 4.0

import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CMAP_LIGHT = mcolors.LinearSegmentedColormap.from_list(
    "lamp_white", ["#ffffff", "#fffde7", "#fff176", "#ffb300", "#e65100", "#b71c1c"]
)

RESULTS_BASE_DIR = f"ablation_results_{GLOBAL_W_SOFT:g}"
RESULTS_DIR = ""
COMBINED_DIR = ""
PAPER_PLOTS_DIR = os.path.join(_SCRIPT_DIR, "paper_plots")
ABLATION_RESULTS_DIR = os.path.join(
    RESULTS_BASE_DIR, "results_v3_multiple_full_bound_sync_kdtree"
)


def configure_results_paths(enable_obstacles: bool = False) -> str:
    """Mirror comparison_code layout: separate dirs for with/without obstacles."""
    global RESULTS_DIR, COMBINED_DIR, ENABLE_OBSTACLES
    folder = (
        "behavior_viz_with_obstacle"
        if enable_obstacles
        else "behavior_viz_wo_obstacle"
    )
    base = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "results", "raw", folder))
    RESULTS_DIR = os.path.join(base, "results")
    COMBINED_DIR = os.path.join(base, "combined")
    ENABLE_OBSTACLES = enable_obstacles
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(COMBINED_DIR, exist_ok=True)
    return RESULTS_DIR


def paper_robots_dir(n_robots: int) -> str:
    """``paper_plots/{n}_robots/`` - one folder per swarm size in ``PAPER_N_LIST``."""
    path = os.path.join(PAPER_PLOTS_DIR, f"{n_robots}_robots")
    os.makedirs(path, exist_ok=True)
    return path


def ensure_paper_dirs() -> list[str]:
    os.makedirs(PAPER_PLOTS_DIR, exist_ok=True)
    return [paper_robots_dir(n) for n in PAPER_N_LIST]


configure_results_paths(enable_obstacles=False)

OUTPUT_CHOICES = ("results", "combined", "paper")


@dataclass(frozen=True)
class OutputSelection:
    results: bool = True
    combined: bool = True
    paper: bool = True

    @classmethod
    def all(cls) -> "OutputSelection":
        return cls()

    @classmethod
    def from_names(cls, names: list[str]) -> "OutputSelection":
        chosen = set(names)
        return cls(
            results="results" in chosen,
            combined="combined" in chosen,
            paper="paper" in chosen,
        )

    def selected_labels(self) -> list[str]:
        labels = []
        if self.results:
            labels.append("results")
        if self.combined:
            labels.append("combined")
        if self.paper:
            labels.append("paper")
        return labels

    def without_paper(self) -> "OutputSelection":
        return OutputSelection(self.results, self.combined, False)

N_LIST = [10, 25, 50, 75, 100]
COMBINED_N_LIST = [25, 50, 75, 100]

V_TARGET = 0.10
LOOKAHEAD_DIST = 0.07
ORBIT_RADIUS = 0.25

class HeadlessSim(_CanonicalHeadlessSim):
    def __init__(self, *args, **kwargs):
        if kwargs.get("enable_obstacles") is None:
            kwargs["enable_obstacles"] = ENABLE_OBSTACLES
        kwargs.setdefault("enable_collision", ENABLE_COLLISION)
        kwargs.setdefault("steer_decay", COLLISION_STEER_DECAY)
        kwargs.setdefault("flash_frames", COLLISION_FLASH_FRAMES)
        super().__init__(*args, **kwargs)


def run_ablation_study():
    results = []
    
    cases = [
        {"name": "Only Seperation",   "q_coh": 0.0, "q_ali": 0.0, "w_soft": 1e7},
        {"name": "Only Cohesion",   "q_coh": GLOBAL_Q_COH, "q_ali": 0.0, "w_soft": 0.0},
        {"name": "Only Alignment",  "q_coh": 0.0, "q_ali": GLOBAL_Q_ALI,  "w_soft": 0.0},
        {"name": "No Seperation",   "q_coh": GLOBAL_Q_COH, "q_ali": GLOBAL_Q_ALI, "w_soft": 0.0},
        {"name": "No Cohesion",   "q_coh": 0.0, "q_ali": GLOBAL_Q_ALI, "w_soft": 1e7},
        {"name": "No Alignment",  "q_coh": GLOBAL_Q_COH, "q_ali": 0.0,  "w_soft": 1e7},
        {"name": "No Reynolds Rule (RR)",   "q_coh": 0.0, "q_ali": 0.0,  "w_soft": 0.0, "w_soft_obs": 0.0},

        {"name": "All RR with Very Weak (w_soft=1e3)", "q_coh": GLOBAL_Q_COH, "q_ali": GLOBAL_Q_ALI, "w_soft": 1e3},
        {"name": "All RR with Weak Separation (w_soft=1e5)", "q_coh": GLOBAL_Q_COH, "q_ali": GLOBAL_Q_ALI, "w_soft": 1e5},
        {"name": "All RR with Medium Separation (w_soft=1e6)", "q_coh": GLOBAL_Q_COH, "q_ali": GLOBAL_Q_ALI, "w_soft": 1e6},      
        {"name": "All RR with Nominal Separation (w_soft=1e7)", "q_coh": GLOBAL_Q_COH, "q_ali": GLOBAL_Q_ALI, "w_soft": 1e7},
        {"name": "All RR with Strong Separation (w_soft=1e9)", "q_coh": GLOBAL_Q_COH, "q_ali": GLOBAL_Q_ALI, "w_soft": 1e9},
        {"name": "All RR with Very Strong Separation (w_soft=1e15)", "q_coh": GLOBAL_Q_COH, "q_ali": GLOBAL_Q_ALI, "w_soft": 1e15},
        {"name": "All RR with Extremely Strong Separation (w_soft=1e25)", "q_coh": GLOBAL_Q_COH, "q_ali": GLOBAL_Q_ALI, "w_soft": 1e25},

    ]
    
    os.makedirs(ABLATION_RESULTS_DIR, exist_ok=True)
    
    for case in cases:
        print(f"Running Case: {case['name']}...")
        sim = HeadlessSim(n_robots=N_LIST[0], behavior_mode=SwarmBehavior.PHOTOTAXIS, 
                          q_coh=case["q_coh"], q_ali=case["q_ali"], w_soft=case["w_soft"],
                          w_soft_obs=case.get("w_soft_obs", GLOBAL_W_SOFT_OBS))
        sim.run(MAX_STEPS)
        
        start_idx = MAX_STEPS // 2
        min_dist_overall = np.min(sim.min_dist_log)
        mean_avg_min_sep = np.mean(sim.avg_min_sep_log[start_idx:])
        mean_compactness = np.mean(sim.mean_dist_log[start_idx:])
        mean_coherence = np.mean(sim.coherence_log[start_idx:])
        
        total_control_steps = sim.n_robots * MAX_STEPS
        total_failures = sum(sim.failure_counts)
        fail_rate = (total_failures / total_control_steps) * 100 if total_control_steps > 0 else 0.0

        results.append({
            "Variant": case["name"],
            "Min Dist (m)": f"{min_dist_overall:.3f}",
            "Avg Min Dist (m)": f"{mean_avg_min_sep:.3f}",
            "Compactness (m)": f"{mean_compactness:.3f}",
            "Coherence (0-1)": f"{mean_coherence:.3f}",
            "Failures (%)": f"{fail_rate:.1f}%",
        })
        
    print("\n--- Ablation Results ---")
    keys = ["Variant", "Min Dist (m)", "Avg Min Dist (m)", "Compactness (m)", "Coherence (0-1)", "Failures (%)"]
    print(f"{keys[0]:<28} {keys[1]:<12} {keys[2]:<16} {keys[3]:<12} {keys[4]:<12} {keys[5]:<12}")
    for r in results:
        print(f"{r['Variant']:<28} {r['Min Dist (m)']:<12} {r['Avg Min Dist (m)']:<16} {r['Compactness (m)']:<12} {r['Coherence (0-1)']:<12} {r['Failures (%)']:<12}")
        
    with open(os.path.join(ABLATION_RESULTS_DIR, "ablation_table.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    
    plot_ablation_metrics_summary(results)

def plot_ablation_metrics_summary(results):
    variants = [r["Variant"] for r in results]
    fail_rates = [float(r["Failures (%)"].replace('%', '')) for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.bar(variants, fail_rates, color='#ef4444', alpha=0.8, edgecolor='#7f1d1d')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.ylabel("Solver Failure Rate (%)", fontsize=10)
    plt.title("Reliability Analysis: Aggregate Solver Failures per Variant", fontsize=12, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(ABLATION_RESULTS_DIR, "ablation_failures.png"), dpi=200)
    plt.close()

def run_persistence_study():
    """
    Evaluate swarm stability when operating with the Collective Transport behavior
    (no global light-field attractor).  Compares full Reynolds vs. no-Reynolds
    to highlight the contribution of cohesion/alignment/separation.
    """
    print("\n--- Running Persistence Study (Collective Transport, No Light Source) ---")
    results = []

    cases = [
        {"name": "Full RR (Transport)",   "q_coh": GLOBAL_Q_COH, "q_ali": GLOBAL_Q_ALI, "w_soft": GLOBAL_W_SOFT},
        {"name": "No Reynolds (Transport)", "q_coh": 0.0,          "q_ali": 0.0,          "w_soft": 0.0, "w_soft_obs": 0.0},
    ]

    for case in cases:
        print(f"Running: {case['name']}...")
        sim = HeadlessSim(
            n_robots=N_LIST[0],
            behavior_mode=SwarmBehavior.NO_LIGHT,
            q_coh=case["q_coh"], q_ali=case["q_ali"], w_soft=case["w_soft"],
            w_soft_obs=case.get("w_soft_obs", GLOBAL_W_SOFT_OBS),
        )

        group_radius_history = []
        min_dist_history     = []
        avg_min_sep_history  = []
        for _ in range(MAX_STEPS):
            sim.step()
            positions = np.array([r.pos for r in sim.robots])
            centroid  = np.mean(positions, axis=0)
            max_dist_from_center = np.max(np.linalg.norm(positions - centroid, axis=1))
            group_radius_history.append(max_dist_from_center)
            min_dist_history.append(sim.min_dist_log[-1])
            avg_min_sep_history.append(sim.avg_min_sep_log[-1])

        final_radius     = group_radius_history[-1]
        overall_min_dist = np.min(min_dist_history)
        mean_avg_min_sep = np.mean(avg_min_sep_history[MAX_STEPS // 2:])

        results.append({
            "Variant":        case["name"],
            "Min Dist (m)":   f"{overall_min_dist:.3f}",
            "Avg Min Dist (m)": f"{mean_avg_min_sep:.3f}",
            "Final Radius (m)": f"{final_radius:.3f}",
            "Status": "Coherent" if final_radius < 0.3 else "Scattered",
        })

    print("\n--- Persistence Results ---")
    keys = ["Variant", "Min Dist (m)", "Avg Min Dist (m)", "Final Radius (m)", "Status"]
    print(f"{keys[0]:<25} {keys[1]:<12} {keys[2]:<16} {keys[3]:<16} {keys[4]:<12}")
    for r in results:
        print(f"{r['Variant']:<25} {r['Min Dist (m)']:<12} {r['Avg Min Dist (m)']:<16} {r['Final Radius (m)']:<16} {r['Status']:<12}")

    with open(os.path.join(ABLATION_RESULTS_DIR, "persistence_table.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

def _collect_snapshots(configs, steps=SNAPSHOT_STEPS):
    """Run simulations and return (snaps_list, labels) for each config row."""
    snaps_list = []
    labels = []
    for cfg in configs:
        sim = cfg["sim_func"]()
        snaps = []
        current_step = 0
        for target_step in steps:
            while current_step < target_step:
                sim.step()
                current_step += 1
            snaps.append(sim.get_snapshot())
        snaps_list.append(snaps)
        labels.append(cfg["label"])
    return snaps_list, labels


def _combined_case_configs(n_robots: int) -> list[dict]:
    """Six behavior cases for one combined figure at fixed ``n_robots``."""

    def no_light():
        sim = HeadlessSim(n_robots=n_robots, behavior_mode=SwarmBehavior.PHOTOTAXIS)
        sim.field = None
        return sim

    return [
        {
            "label": "Phototaxis",
            "sim_func": lambda: HeadlessSim(
                n_robots=n_robots, behavior_mode=SwarmBehavior.PHOTOTAXIS,
            ),
        },
        {
            "label": "Phototaxis Orbital",
            "sim_func": lambda: HeadlessSim(
                n_robots=n_robots,
                behavior_mode=SwarmBehavior.PHOTOTAXIS_ORBITAL,
            ),
        },
        {
            "label": "Phototaxis Orbital Contracting",
            "sim_func": lambda: HeadlessSim(
                n_robots=n_robots,
                behavior_mode=SwarmBehavior.PHOTOTAXIS_ORBITAL_CONTRACTING,
            ),
        },
        {
            "label": "Collective Transport",
            "sim_func": lambda: HeadlessSim(
                n_robots=n_robots, behavior_mode=SwarmBehavior.NO_LIGHT,
            ),
        },
        {
            "label": "Phototaxis w/o Reynolds",
            "sim_func": lambda: HeadlessSim(
                n_robots=n_robots,
                behavior_mode=SwarmBehavior.PHOTOTAXIS,
                q_coh=0.0,
                q_ali=0.0,
                w_soft=0.0,
                w_soft_obs=0.0,
            ),
        },
        {
            "label": "No light",
            "sim_func": no_light,
        },
    ]


def generate_combined_nb_figures(steps=SNAPSHOT_STEPS):
    """
    One PNG per ``COMBINED_N_LIST`` entry: 6 behavior rows × 5 time columns at fixed N_b.
    """
    os.makedirs(COMBINED_DIR, exist_ok=True)
    obstacle_note = "with static obstacles" if ENABLE_OBSTACLES else "without obstacles"

    for n in COMBINED_N_LIST:
        print(f"Running combined behavior cases at $N_b={n}$ …")
        configs = _combined_case_configs(n)
        snaps_list, labels = _collect_snapshots(configs, steps=steps)
        subfigure_row_indices = {row: row for row in range(len(labels))}
        title = (
            f"Swarm Behaviors ($N_b = {n}$, {obstacle_note}) - "
            f"LA Slack Dt-CBF ($d_{{safe}}={D_SAFE*100:.0f}$ cm)"
        )
        _save_publication_snapshot_grid(
            snaps_list,
            labels,
            title,
            os.path.join(COMBINED_DIR, f"combined_nb_{n}"),
            steps=steps,
            subfigure_row_indices=subfigure_row_indices,
            is_final=True,
            log_label=f"combined_nb_{n}",
            export_formats=("png",),
            show_obstacles_in_legend=ENABLE_OBSTACLES,
        )


def _snapshot_grid_figsize(n_rows: int, n_cols: int) -> tuple[float, float]:
    """IEEE-friendly panel sizing - mirrors ``_combined_collision_figsize``."""
    n_cols = max(n_cols, 1)
    n_rows = max(n_rows, 1)
    panel_w = max(2.5, min(3.2, 16.0 / n_cols))
    panel_h = panel_w
    return panel_w * n_cols + 0.55, panel_h * n_rows + 1.30


def _subplot_block_center_x(axes) -> float:
    """Horizontal center of a subplot grid in figure coordinates."""
    ax_lo = axes[0, 0].get_position()
    ax_hi = axes[-1, -1].get_position()
    return 0.5 * (ax_lo.x0 + ax_hi.x1)


def _time_label(step: int) -> str:
    return f"$t = {int(round(step * DT))}\\,\\mathrm{{s}}$"


def _grid_panel_axis_labels(*, row: int, col: int, n_rows: int, steps, step_idx: int):
    """Shared axis labels: y on col 0 of every row; x on bottom row."""
    return {
        "col_title": _time_label(steps[step_idx]) if row == 0 else "",
        "row_ylabel": "",
        "xlabel": "Position $x$ [m]" if row == n_rows - 1 else "",
        "ylabel": "Position $y$ [m]" if col == 0 else "",
    }


def _style_arena_panel(
    ax,
    *,
    col_title: str = "",
    row_ylabel: str = "",
    xlabel: str = "",
    ylabel: str = "",
):
    """Publication arena axes - serif labels, minimal spines."""
    ax.set_aspect("equal")
    ax.set_xlim(-WALL_HALF * 1.05, WALL_HALF * 1.05)
    ax.set_ylim(-WALL_HALF * 1.05, WALL_HALF * 1.05)
    ax.set_facecolor(PANEL_BG_COLOR)
    ax.set_title(
        col_title,
        fontsize=FONTSIZE_PANEL_TITLE,
        fontweight="bold",
        pad=6,
        color="#111111",
    )
    if row_ylabel:
        ax.set_ylabel(row_ylabel, fontsize=FONTSIZE_AXIS, labelpad=10, color="#222222")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONTSIZE_AXIS, labelpad=6, color="#222222")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONTSIZE_AXIS, labelpad=4, color="#222222")
    ax.tick_params(colors="#333333", labelsize=max(FONTSIZE_TICK - 1, 6), length=2, width=0.6)
    for spine in ax.spines.values():
        spine.set_color("#888888")
        spine.set_linewidth(0.6)


def _draw_robot_body_and_arrows(ax, pos, theta, color, edge_col="#222222", edge_lw=0.7):
    """Robot ellipse + small white heading arrow (head and tail)."""
    ax.add_patch(
        Ellipse(
            xy=pos, width=ROBOT_A * 2, height=ROBOT_B * 2, angle=np.rad2deg(theta),
            facecolor=color, edgecolor=edge_col, linewidth=edge_lw,
            alpha=0.92, zorder=6,
        )
    )

    orient_end = pos + HEADING_ARROW_LENGTH * np.array([np.cos(theta), np.sin(theta)])
    ax.annotate(
        "", xy=orient_end, xytext=pos,
        arrowprops=dict(arrowstyle="->", color="white", lw=HEADING_ARROW_LW,
                        mutation_scale=HEADING_ARROW_HEAD, alpha=0.9),
        zorder=7,
    )


def _draw_snapshot_content(ax, snap):
    """Draw arena content (robots, field, obstacles, collisions) on styled axes."""
    ax.add_patch(
        plt.Rectangle(
            (-WALL_HALF, -WALL_HALF), 2 * WALL_HALF, 2 * WALL_HALF,
            linewidth=1.0, edgecolor=ARENA_EDGE_COLOR, facecolor="none",
            linestyle="-", zorder=2,
        )
    )

    if snap["heatmap"] is not None:
        ax.imshow(
            snap["heatmap"], origin="lower", extent=snap["extent"],
            cmap=CMAP_LIGHT, alpha=0.55, zorder=1, interpolation="bilinear",
        )
        for src in snap["field_sources"]:
            ax.plot(*src, "o", color="#b45309", markersize=4.5, alpha=0.95, zorder=3)

    positions = snap["positions"]
    headings = snap["headings"]
    overlap_pairs = snap.get("overlap_pairs", [])
    robot_obstacle_overlaps = snap.get("robot_obstacle_overlaps", [])
    obstacles = snap.get("obstacles", [])
    colliding = set(snap.get("colliding_indices", []))
    flash_alphas = snap.get("flash_alphas", [0.0] * len(positions))

    colliding_robots = set(colliding)
    for i, j in overlap_pairs:
        colliding_robots.add(i)
        colliding_robots.add(j)
    for i, _obs_idx in robot_obstacle_overlaps:
        colliding_robots.add(i)

    for idx, (pos, th) in enumerate(zip(positions, headings)):
        color = PUBLICATION_ROBOT_COLOR
        in_collision = idx in colliding_robots
        edge_col = VIOL_COLOR if in_collision else "#222222"
        edge_lw = 1.4 if in_collision else 0.7
        _draw_robot_body_and_arrows(
            ax, pos, th, color,
            edge_col=edge_col, edge_lw=edge_lw,
        )

        flash_alpha = flash_alphas[idx] if idx < len(flash_alphas) else 0.0
        if flash_alpha > 0.0:
            ax.add_patch(
                Ellipse(
                    xy=pos,
                    width=(ROBOT_A * 2) * (1.0 + 0.5 * flash_alpha),
                    height=(ROBOT_B * 2) * (1.0 + 0.5 * flash_alpha),
                    angle=np.rad2deg(th),
                    facecolor="none", edgecolor=VIOL_COLOR,
                    linewidth=1.2 * flash_alpha, alpha=0.85 * flash_alpha, zorder=8,
                )
            )

    for obs in obstacles:
        is_moving = np.linalg.norm(obs["vel"]) > 1e-6
        face_col = "#FF0000"
        edge_col = "#FF0000"
        ax.add_patch(
            Circle(
                obs["pos"], obs["radius"],
                facecolor=face_col, edgecolor=edge_col,
                linewidth=1.0, alpha=0.88, zorder=5,
            )
        )



def _snapshot_legend_handles(show_obstacles: bool = False, show_light: bool = True):
    """Shared legend entries for publication snapshot figures."""
    handles = [
        Line2D(
            [0], [0], color=ARENA_EDGE_COLOR, lw=1.0, linestyle="-",
            label="World boundary",
        ),
    ]
    if show_light:
        handles.append(
            Patch(
                facecolor="#ffb300", edgecolor="#e65100", linewidth=0.6,
                alpha=0.75, label="Light",
            )
        )
    handles.append(
        Patch(facecolor=PUBLICATION_ROBOT_COLOR, edgecolor="#222222", linewidth=0.6, label="Robot"),
    )
    if show_obstacles:
        handles.append(
            Patch(facecolor="#FF0000", edgecolor="#FF0000", linewidth=0.8, label="Static obstacle")
        )
    if ENABLE_COLLISION:
        handles.append(
            Patch(
                facecolor=PUBLICATION_ROBOT_COLOR, edgecolor=VIOL_COLOR,
                linewidth=1.4, label="Active collision",
            )
        )
    return handles


def _export_figure(fig, output_basename, formats, is_final=False, log_label=None):
    """Write figure assets. ``results/`` and ``combined/`` use PNG; ``paper_plots/`` uses PNG + PDF."""
    saved = []
    if "svg" in formats:
        svg_path = f"{output_basename}.svg"
        fig.savefig(svg_path, bbox_inches="tight", facecolor="white", format="svg")
        saved.append(os.path.basename(svg_path))
    if "png" in formats:
        png_path = f"{output_basename}.png"
        fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
        saved.append(os.path.basename(png_path))
    if "pdf" in formats:
        pdf_path = f"{output_basename}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        saved.append(os.path.basename(pdf_path))
    if is_final and log_label and saved:
        print(f"  ├── Saved: {', '.join(saved)} ({log_label})")
    plt.close(fig)


def _save_publication_snapshot_grid(
    snaps_list,
    labels,
    suptitle,
    output_basename,
    steps=SNAPSHOT_STEPS,
    subfigure_row_indices=None,
    is_final=True,
    log_label=None,
    export_formats=("png",),
    show_obstacles_in_legend: bool | None = None,
    show_light_in_legend: bool = True,
):
    """Save snapshot grid. Pass ``export_formats=('png', 'pdf')`` for paper exports."""
    if show_obstacles_in_legend is None:
        show_obstacles_in_legend = ENABLE_OBSTACLES
    n_rows = len(snaps_list)
    n_cols = len(steps)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=_snapshot_grid_figsize(n_rows, n_cols),
        facecolor="white",
        squeeze=False,
    )
    left_margin = max(0.10, 0.06 + 0.007 * n_cols)

    for row, (snaps, row_label) in enumerate(zip(snaps_list, labels)):
        for col, snap in enumerate(snaps):
            ax = axes[row, col]
            _style_arena_panel(
                ax,
                **_grid_panel_axis_labels(
                    row=row, col=col, n_rows=n_rows, steps=steps, step_idx=col,
                ),
            )
            _draw_snapshot_content(ax, snap)
            if subfigure_row_indices is not None and col == 0:
                tag_idx = subfigure_row_indices.get(row)
                if tag_idx is not None:
                    add_subfigure_label(ax, tag_idx, x=-0.16)

    legend_handles = _snapshot_legend_handles(
        show_obstacles=show_obstacles_in_legend,
        show_light=show_light_in_legend,
    )
    legend_labels = [h.get_label() for h in legend_handles]
    fig.suptitle(
        suptitle,
        fontsize=FONTSIZE_SUPTITLE,
        fontweight="bold",
        y=0.98,
        color="#111111",
    )
    fig.tight_layout(rect=[left_margin, 0.08, 1, 0.93])
    add_figure_legend(
        fig,
        handles=legend_handles,
        labels=legend_labels,
        x=_subplot_block_center_x(axes),
        y=0.015,
        single_row=True,
    )
    _export_figure(
        fig,
        output_basename,
        export_formats,
        is_final=is_final,
        log_label=log_label or f"{n_rows}×{n_cols}",
    )


def plot_snapshots_comparison(
    configs,
    title,
    filename,
    steps=SNAPSHOT_STEPS,
    output_dir=None,
    return_snapshots=False,
    show_light_in_legend: bool = True,
):
    """configs: list of {"label": str, "sim_func": callable}"""
    print(f"Running comparison simulations for {filename}...")
    output_dir = output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    snaps_list, labels = _collect_snapshots(configs, steps=steps)
    basename = os.path.join(output_dir, os.path.splitext(filename)[0])
    _save_publication_snapshot_grid(
        snaps_list, labels, title, basename, steps=steps,
        is_final=True, log_label=filename,
        show_light_in_legend=show_light_in_legend,
    )

    if return_snapshots:
        return snaps_list, labels
    return None


def _collect_sim_snapshots(sim, steps=SNAPSHOT_STEPS):
    """Advance ``sim`` and return arena snapshots at selected steps."""
    snaps = []
    current_step = 0
    for target_step in steps:
        while current_step < target_step:
            sim.step()
            current_step += 1
        snaps.append(sim.get_snapshot())
    return snaps


PAPER_OBSTACLE_ROWS = (
    (r"$N_b = 50$, without obstacles", False),
    (r"$N_b = 50$, with static obstacles", True),
)


def _generate_paper_obstacle_comparison_figure(
    basename: str,
    sim_factory,
    n_robots: int,
    output_dir: str,
    steps=SNAPSHOT_STEPS,
    show_light_in_legend: bool = True,
):
    """
    Two-row paper figure: time evolution at ``n_robots`` without / with static obstacles.
    No figure title - manuscript panels use subfigure labels only.
    """
    n_rows = len(PAPER_OBSTACLE_ROWS)
    n_cols = len(steps)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=_snapshot_grid_figsize(n_rows, n_cols),
        facecolor="white",
        squeeze=False,
    )
    left_margin = 0.12

    for row_idx, (row_label, with_obstacles) in enumerate(PAPER_OBSTACLE_ROWS):
        print(f"  Paper figure '{basename}' @ N={n_robots}: {row_label} …")
        sim = sim_factory(with_obstacles)
        snaps = _collect_sim_snapshots(sim, steps=steps)
        for col, snap in enumerate(snaps):
            ax = axes[row_idx, col]
            _style_arena_panel(
                ax,
                **_grid_panel_axis_labels(
                    row=row_idx, col=col, n_rows=n_rows, steps=steps, step_idx=col,
                ),
            )
            _draw_snapshot_content(ax, snap)
            if col == 0:
                add_subfigure_label(ax, row_idx, x=-0.18)

    legend_handles = _snapshot_legend_handles(
        show_obstacles=True,
        show_light=show_light_in_legend,
    )
    legend_labels = [h.get_label() for h in legend_handles]
    fig.tight_layout(rect=[left_margin, 0.08, 1, 0.98])
    add_figure_legend(
        fig,
        handles=legend_handles,
        labels=legend_labels,
        x=_subplot_block_center_x(axes),
        y=0.015,
        single_row=True,
    )
    filename = f"{basename}_with_{n_robots}_robots"
    out_base = os.path.join(output_dir, filename)
    _export_figure(
        fig, out_base, ("png", "pdf"),
        is_final=True, log_label=f"{filename} 2×{n_cols}",
    )


def _phototaxis_sim_factory(n_robots: int):
    def factory(with_obstacles: bool):
        return HeadlessSim(
            n_robots=n_robots,
            behavior_mode=SwarmBehavior.PHOTOTAXIS,
            enable_obstacles=with_obstacles,
        )
    return factory


def _orbital_sim_factory(n_robots: int):
    def factory(with_obstacles: bool):
        return HeadlessSim(
            n_robots=n_robots,
            behavior_mode=SwarmBehavior.PHOTOTAXIS_ORBITAL,
            enable_obstacles=with_obstacles,
        )
    return factory


def _orbital_contracting_sim_factory(n_robots: int):
    def factory(with_obstacles: bool):
        return HeadlessSim(
            n_robots=n_robots,
            behavior_mode=SwarmBehavior.PHOTOTAXIS_ORBITAL_CONTRACTING,
            enable_obstacles=with_obstacles,
        )
    return factory


def _no_reynolds_sim_factory(n_robots: int):
    def factory(with_obstacles: bool):
        return HeadlessSim(
            n_robots=n_robots,
            behavior_mode=SwarmBehavior.PHOTOTAXIS,
            q_coh=0.0,
            q_ali=0.0,
            w_soft=0.0,
            w_soft_obs=0.0,
            w_soft_lin=0.0,
            w_soft_obs_lin=0.0,
            enable_avoidance=False,
            enable_obstacles=with_obstacles,
        )
    return factory


def _no_light_sim_factory(n_robots: int):
    def factory(with_obstacles: bool):
        sim = HeadlessSim(
            n_robots=n_robots,
            behavior_mode=SwarmBehavior.PHOTOTAXIS,
            enable_obstacles=with_obstacles,
        )
        sim.field = None
        return sim
    return factory


def _generate_all_paper_figures(steps=SNAPSHOT_STEPS):
    """Manuscript figures: each behavior × (wo / with obstacles) × ``PAPER_N_LIST``."""
    ensure_paper_dirs()
    print(
        f"\n=== Generating paper figures -> {PAPER_PLOTS_DIR}/{{N}}_robots/ "
        f"(N={PAPER_N_LIST}, PNG + PDF) ==="
    )

    paper_figure_specs = [
        ("paper_phototaxis", _phototaxis_sim_factory, True),
        ("paper_phototaxis_orbital", _orbital_sim_factory, True),
        ("paper_phototaxis_orbital_contracting", _orbital_contracting_sim_factory, True),
        ("paper_phototaxis_no_reynolds", _no_reynolds_sim_factory, True),
        ("paper_no_light", _no_light_sim_factory, False),
    ]
    for n_robots in PAPER_N_LIST:
        output_dir = paper_robots_dir(n_robots)
        print(f"\n--- N={n_robots} -> {output_dir} ---")
        for basename, factory_builder, show_light in paper_figure_specs:
            _generate_paper_obstacle_comparison_figure(
                basename,
                factory_builder(n_robots),
                n_robots=n_robots,
                output_dir=output_dir,
                steps=steps,
                show_light_in_legend=show_light,
            )


def generate_combined_behavior_figures(
    steps=SNAPSHOT_STEPS,
    outputs: OutputSelection | None = None,
):
    """Build selected outputs: results/combined (PNG). Paper figures are generated separately."""
    outputs = outputs or OutputSelection.all()
    if outputs.results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
    if outputs.combined:
        os.makedirs(COMBINED_DIR, exist_ok=True)

    behavior_sections = [
        {
            "title": "Phototaxis",
            "results_basename": "phototaxis_snapshots",
            "configs": [
                {
                    "label": f"$N_b = {n}$",
                    "sim_func": (
                        lambda n_val=n: HeadlessSim(
                            n_robots=n_val, behavior_mode=SwarmBehavior.PHOTOTAXIS
                        )
                    ),
                }
                for n in N_LIST
            ],
        },
        {
            "title": "Phototaxis Orbital",
            "results_basename": "phototaxis_orbital_snapshots",
            "configs": [
                {
                    "label": f"$N_b = {n}$",
                    "sim_func": (
                        lambda n_val=n: HeadlessSim(
                            n_robots=n_val,
                            behavior_mode=SwarmBehavior.PHOTOTAXIS_ORBITAL,
                        )
                    ),
                }
                for n in N_LIST
            ],
        },
        {
            "title": "Phototaxis Orbital Contracting",
            "results_basename": "phototaxis_orbital_contracting_snapshots",
            "configs": [
                {
                    "label": f"$N_b = {n}$",
                    "sim_func": (
                        lambda n_val=n: HeadlessSim(
                            n_robots=n_val,
                            behavior_mode=SwarmBehavior.PHOTOTAXIS_ORBITAL_CONTRACTING,
                        )
                    ),
                }
                for n in N_LIST
            ],
        },
        {
            "title": "Collective Transport",
            "results_basename": "no_light_snapshots",
            "configs": [
                {
                    "label": f"$N_b = {n}$",
                    "sim_func": (
                        lambda n_val=n: HeadlessSim(
                            n_robots=n_val, behavior_mode=SwarmBehavior.NO_LIGHT
                        )
                    ),
                }
                for n in N_LIST
            ],
        },
    ]

    if outputs.results:
        for section in behavior_sections:
            print(f"Running behavior snapshots: {section['results_basename']} …")
            snaps_list, labels = _collect_snapshots(section["configs"], steps=steps)
            grid_title = f"{section['title']} - LA Slack Dt-CBF ($d_{{safe}}={D_SAFE*100:.0f}$ cm)"
            _save_publication_snapshot_grid(
                snaps_list, labels, grid_title,
                os.path.join(RESULTS_DIR, section["results_basename"]),
                steps=steps, is_final=True,
                log_label=section["results_basename"],
                export_formats=("png",),
                show_light_in_legend=section["title"] != "Collective Transport",
            )

    if outputs.combined:
        generate_combined_nb_figures(steps=steps)

def run_visual_tasks(outputs: OutputSelection | None = None):
    outputs = outputs or OutputSelection.all()
    obstacle_tag = "WITH obstacles" if ENABLE_OBSTACLES else "WITHOUT obstacles"
    print(f"\n=== Generating behavior snapshots ({obstacle_tag}) ===")
    if outputs.results:
        print(f"  results/  -> {RESULTS_DIR}  (PNG)")
    if outputs.combined:
        print(f"  combined/ -> {COMBINED_DIR}  (PNG)")

    if outputs.results or outputs.combined:
        generate_combined_behavior_figures(outputs=outputs)

    if not outputs.results:
        return

    print(f"Generating Comparative Snapshots (No Reynolds across Robot Counts)...")
    plot_snapshots_comparison([
        {
            "label": f"$N_b = {n}$",
            "sim_func": (
                lambda n_val=n: HeadlessSim(
                    n_robots=n_val,
                    behavior_mode=SwarmBehavior.PHOTOTAXIS,
                    q_coh=0.0,
                    q_ali=0.0,
                    w_soft=0.0,
                    w_soft_obs=0.0,
                )
            ),
        }
        for n in N_LIST
    ], "Phototaxis without Reynolds Rules", "phototaxis_no_reynolds_comparison.png")

    run_persistence_visuals()

def run_persistence_visuals():
    print(f"Generating Persistence snapshots (Phototaxis with light OFF, {N_LIST})...")

    def sim_no_light(n):
        sim = HeadlessSim(n_robots=n, behavior_mode=SwarmBehavior.PHOTOTAXIS)
        sim.field = None
        return sim

    plot_snapshots_comparison([
        {"label": f"$N_b = {n}$", "sim_func": (lambda n_val=n: sim_no_light(n_val))}
        for n in N_LIST
    ], "Swarm Persistence without Light Source", "lights_off_with_reynold.png",
       show_light_in_legend=False)

def run_performance():
    print(f"Gathering timing info for {N_LIST}...")
    
    all_solve_times = []
    all_fail_rates = []
    solve_time_stats = []
    for n in N_LIST:
        sim = HeadlessSim(n_robots=n, behavior_mode=SwarmBehavior.PHOTOTAXIS)
        sim.run(100)
        filtered = [[t for t in sim.solve_times[i] if t <= 10000] for i in range(n)]
        all_solve_times.append(filtered)
        
        flat_times = [t for sublist in filtered for t in sublist]
        if flat_times:
            avg_ms = np.mean(flat_times) / 1000.0
            std_ms = np.std(flat_times) / 1000.0
        else:
            avg_ms = 0.0
            std_ms = 0.0
            
        num_steps = len(sim.solve_times[0])
        step_consensus_times_ms = []
        for step in range(num_steps):
            step_sum_us = sum(sim.solve_times[i][step] for i in range(n) if step < len(sim.solve_times[i]))
            step_consensus_times_ms.append(step_sum_us / 1000.0)
            
        if step_consensus_times_ms:
            avg_swarm_ms = np.mean(step_consensus_times_ms)
            std_swarm_ms = np.std(step_consensus_times_ms)
        else:
            avg_swarm_ms = 0.0
            std_swarm_ms = 0.0
            
        solve_time_stats.append((n, avg_ms, std_ms, avg_swarm_ms, std_swarm_ms))
        
        rate = (sum(sim.failure_counts) / (n * 100)) * 100
        all_fail_rates.append(rate)

    num_plots = len(N_LIST)
    fig, axes = plt.subplots(num_plots, 1, figsize=(max(8, 0.4*max(N_LIST)), 5.5*num_plots), squeeze=False)
    
    for i, (n, times, rate) in enumerate(zip(N_LIST, all_solve_times, all_fail_rates)):
        ax = axes[i, 0]
        ax.boxplot(times, labels=[f"R{j}" for j in range(n)])
        ax.set_ylabel(r"Solve time ($\mu$s)")
        ax.set_title(f"Per-Robot Solve Time ($N_b={n}, M_c={MAX_CONSENSUS}$)\nAvg. Solver Failure Rate: {rate:.1f}%")
    
    plt.tight_layout()
    plt.savefig(os.path.join(ABLATION_RESULTS_DIR, "solve_times_box.png"), bbox_inches='tight')
    plt.close()

    print("Running scalability test (N=5, 10, 15)...")
    wall_times = []
    reliability_Ns = [5, 10, 15, 20, 25, 50, 100]
    reliability_fail_rates = []
    for n in reliability_Ns:
        sim = HeadlessSim(n_robots=n, behavior_mode=SwarmBehavior.PHOTOTAXIS)
        
        t0 = time.time()
        sim.run(50)
        t1 = time.time()
        
        avg_wall_time_per_step = (t1 - t0) / 50.0 * 1000
        wall_times.append(avg_wall_time_per_step)
        
        rate = (sum(sim.failure_counts) / (n * 50)) * 100
        reliability_fail_rates.append(rate)
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    ax1.plot(reliability_Ns, wall_times, marker='o', lw=2, color='#2563eb')
    ax1.axhline(100, color='#ef4444', linestyle='--', alpha=0.6, label='10 Hz Budget (100ms)')
    ax1.axhline(33.33, color='#94a3b8', linestyle='--',  alpha=0.8, label='30 Hz Target (33.3ms)')
    ax1.set_xlabel("Number of Robots ($N_b$)")
    ax1.set_ylabel("Avg Wall Time per Step (ms)")
    ax1.set_title("Scalability of TC-DMPC")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(reliability_Ns, reliability_fail_rates, marker='s', lw=2, color='#db2777')
    ax2.set_xlabel("Number of Robots ($N_b$)")
    ax2.set_ylabel("Aggregate Failure Rate (%)")
    ax2.set_title("Reliability scaling vs Swarm Density")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ABLATION_RESULTS_DIR, "performance_scaling.png"), bbox_inches='tight')
    plt.close()
    
    with open(os.path.join(ABLATION_RESULTS_DIR, "scalability.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N_b", "Wall_Time_ms"])
        for n, t in zip(reliability_Ns, wall_times):
            writer.writerow([n, t])

    with open(os.path.join(ABLATION_RESULTS_DIR, "reliability.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N_b", "Failure_Rate_pct"])
        for n, r in zip(reliability_Ns, reliability_fail_rates):
            writer.writerow([n, r])

    with open(os.path.join(ABLATION_RESULTS_DIR, "solve_time_stats.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N_b", "Avg_Solve_Time_ms", "Std_Solve_Time_ms", "Avg_Swarm_Consensus_Time_ms", "Std_Swarm_Consensus_Time_ms"])
        for n, avg_ms, std_ms, avg_swarm_ms, std_swarm_ms in solve_time_stats:
            writer.writerow([n, f"{avg_ms:.3f}", f"{std_ms:.3f}", f"{avg_swarm_ms:.3f}", f"{std_swarm_ms:.3f}"])

def run_for_obstacle_mode(
    enable_obstacles: bool,
    run_full: bool = False,
    studies_only: bool = False,
    outputs: OutputSelection | None = None,
):
    """Configure paths, optionally run ablation/performance, then paper visuals.

    studies_only=True runs the ablation/persistence/timing studies and skips all
    figure generation (it implies the full studies regardless of run_full).
    """
    outputs = outputs or OutputSelection.all()
    out_dir = configure_results_paths(enable_obstacles=enable_obstacles)
    tag = "with static obstacles" if enable_obstacles else "without obstacles"
    folders = "studies only (no figures)" if studies_only else ', '.join(outputs.selected_labels())
    print(f"\n{'=' * 72}\n  Obstacle mode: {tag}\n  Output: {out_dir}\n  Folders: {folders}\n{'=' * 72}")

    run_full = run_full or studies_only
    if run_full:
        run_ablation_study()
        run_persistence_study()
    if not studies_only:
        run_visual_tasks(outputs=outputs)
    if run_full:
        run_performance()

    print(f"Completed ({tag}).")
    if not studies_only and outputs.results:
        print(f"  Individual figures: {RESULTS_DIR}  (PNG)")
    if not studies_only and outputs.combined:
        print(f"  Combined figures:   {COMBINED_DIR}  (PNG)")
    if run_full:
        print(f"  Ablation/perf:      {ABLATION_RESULTS_DIR}")


def run_paper_outputs(steps=SNAPSHOT_STEPS):
    _generate_all_paper_figures(steps=steps)
    subdirs = ", ".join(f"{n}_robots" for n in PAPER_N_LIST)
    print(f"  Paper figures:      {PAPER_PLOTS_DIR}/{{{subdirs}}}  (PNG + PDF)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate swarm behavior snapshot figures for the research paper."
    )
    obstacle_group = parser.add_mutually_exclusive_group()
    obstacle_group.add_argument(
        "--obstacles",
        action="store_true",
        help="Enable static arena obstacles (moving obstacles remain disabled).",
    )
    obstacle_group.add_argument(
        "--wo-obstacles",
        action="store_true",
        help="Disable all obstacles (default).",
    )
    obstacle_group.add_argument(
        "--both",
        action="store_true",
        help=(
            "Generate results/combined for both with- and without-obstacle modes "
            "(does not write paper_plots/)."
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also run ablation, persistence, and performance studies (slow).",
    )
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Run ONLY the ablation, persistence, and timing studies; skip all figures.",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        choices=OUTPUT_CHOICES,
        default=list(OUTPUT_CHOICES),
        metavar="FOLDER",
        help=(
            "Output folders to generate (default: all). "
            "Example: --outputs paper  |  --outputs results combined"
        ),
    )
    args = parser.parse_args()
    outputs = OutputSelection.from_names(args.outputs)

    if args.full_only:
        obstacle_modes = (False, True) if args.both else (bool(args.obstacles),)
        for enable_obs in obstacle_modes:
            run_for_obstacle_mode(
                enable_obstacles=enable_obs,
                studies_only=True,
                outputs=outputs.without_paper(),
            )
        return

    if args.both:
        if outputs.results or outputs.combined or args.full:
            for enable_obs in (False, True):
                run_for_obstacle_mode(
                    enable_obstacles=enable_obs,
                    run_full=args.full,
                    outputs=outputs.without_paper(),
                )
    elif outputs.results or outputs.combined or args.full:
        enable_obs = bool(args.obstacles)
        run_for_obstacle_mode(
            enable_obstacles=enable_obs,
            run_full=args.full,
            outputs=outputs.without_paper(),
        )

    if outputs.paper and not args.both:
        run_paper_outputs()


if __name__ == "__main__":
    main()
