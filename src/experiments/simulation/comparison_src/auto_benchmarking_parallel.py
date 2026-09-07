#!/usr/bin/env python3
"""The benchmark harness: runs every formulation across swarm sizes and collects metrics."""

import os
import sys
import time
import subprocess
import contextlib
import io
import importlib.util
import multiprocessing
import pickle
import re
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from comparison_config import (
    N_LIST, SIM_TIME_SEC, DT, STEPS, STEADY_STATE_STEPS, CASES, COLORS, SEEDS,
    PLOT_DPI, FONTSIZE_SUPTITLE,
    configure_publication_style,
    apply_premium_plot_style, add_figure_legend, save_publication_figure,
    add_subfigure_label,
)

configure_publication_style()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))

SAFETY_LEVEL_FOLDERS = [
    ("src/deterministic_bristlebot_swarm/01_hard_bf", "Hard BF"),
    ("src/deterministic_bristlebot_swarm/02_hard_dt_cbf", "Hard Dt-CBF"),
    ("src/deterministic_bristlebot_swarm/03_slacked_bf", "Slack BF"),
    ("src/deterministic_bristlebot_swarm/04_slacked_dt_cbf", "Slack Dt-CBF"),
    ("src/deterministic_bristlebot_swarm/05_slacked_dt_cbf_with_lookahead", "LA Slack Dt-CBF"),
    ("src/deterministic_bristlebot_swarm/06_slacked_dt_hocbf", "Slack Dt-HOCBF"),
]

def configure_results_paths(enable_obstacles: bool = False) -> str:
    """Set module-level output dirs from obstacle mode; return ``RESULTS_DIR``."""
    global RESULTS_DIR, CSV_DIR
    folder = (
        "results_parallel_with_obstacles"
        if enable_obstacles
        else "results_parallel_wo_obstacles"
    )
    RESULTS_DIR = os.path.join(SCRIPT_DIR, folder)
    CSV_DIR = os.path.join(RESULTS_DIR, "csv_data")
    FIG_DIR = RESULTS_DIR
    return RESULTS_DIR


configure_results_paths(enable_obstacles=False)

SAVE_CSV = True
SAVE_PLOTS = True

def compile_solvers():
    print("=" * 80)
    print("  STEP 1: Regenerating Acados C-code solvers (sequential)")
    print("=" * 80)

    for folder, label in SAFETY_LEVEL_FOLDERS:
        folder_path = os.path.join(ROOT_DIR, folder)
        expert_src_path = os.path.join(folder_path, "expert_src")

        if not os.path.exists(folder_path):
            print(f"  [SKIPPED] {label} - folder not found")
            continue

        print(f"  Compiling: {label:32s} ...", end="", flush=True)
        start_t = time.time()
        try:
            subprocess.run(
                [sys.executable, "generate_solver_via_acados.py"],
                cwd=expert_src_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            print(f" [SUCCESS] ({time.time() - start_t:.1f}s)")
        except subprocess.CalledProcessError as e:
            print(f" [FAILED] ({time.time() - start_t:.1f}s)")
            print(e.stderr if e.stderr else e.stdout)
            sys.exit(1)
    print("\nAll solvers compiled.\n")

def sim_worker_task(args):
    """
    Run one simulation configuration and extract all six telemetry metrics.

    ``args`` is ``(folder, label, n_robots, case_name)``, optionally followed by
    ``enable_obstacles`` (bool, default False) and ``seed`` (int, default 0).
    """
    seed = 0
    if len(args) == 6:
        folder, label, n_robots, case_name, enable_obstacles, seed = args
    elif len(args) == 5:
        folder, label, n_robots, case_name, enable_obstacles = args
    else:
        folder, label, n_robots, case_name = args
        enable_obstacles = False
    folder_path = os.path.join(ROOT_DIR, folder)
    sim_engine_dir = os.path.join(folder_path, "sim_engine")
    
    orig_sys_path = list(sys.path)
    sys.path.insert(0, folder_path)
    sys.path.insert(0, sim_engine_dir)
    
    sim_path = os.path.join(sim_engine_dir, "headless_sim_engine.py")
    if not os.path.isfile(sim_path):
        sys.path = orig_sys_path
        return {"success": False, "error": "No headless sim engine found"}
    
    try:
        spec = importlib.util.spec_from_file_location("dynamic_sim_module", sim_path)
        module = importlib.util.module_from_spec(spec)
        
        for key in list(sys.modules.keys()):
            if any(name in key for name in ["dynamic_sim_module", "headless_sim_engine", "swarm_dynamics", "light_source_model", "obstacle_model", "collision_response", "expert_src"]):
                del sys.modules[key]
                
        spec.loader.exec_module(module)
        
        module.SPAWN_SEED = seed

        collision_history = []
        feasibility_statuses = []
        solve_times_us = []
        steady_state_distances = []
        steady_state_polarizations = []
        robot_jerks = []
        
        with contextlib.redirect_stdout(io.StringIO()):
            sim = module.HeadlessSim(
                n_robots=n_robots,
                behavior_mode=module.SwarmBehavior(case_name),
                enable_obstacles=enable_obstacles,
                enable_collision=True,
            )
            for step_idx in range(STEPS):
                sim.step()
                
                collision_history.append(len(sim.active_collisions))
                
                feasibility_statuses.append(list(sim.solver_statuses))
                
                if step_idx >= (STEPS - STEADY_STATE_STEPS):
                    src_pos = np.zeros(2)
                    if hasattr(sim, 'field') and hasattr(sim.field, 'sources') and sim.field.sources:
                        src_pos = sim.field.sources[0].pos.copy()
                    for r in sim.robots:
                        steady_state_distances.append(np.linalg.norm(r.pos - src_pos))
                        
                    sum_cos = sum(np.cos(r.theta) for r in sim.robots)
                    sum_sin = sum(np.sin(r.theta) for r in sim.robots)
                    polarization = np.sqrt(sum_cos**2 + sum_sin**2) / n_robots
                    steady_state_polarizations.append(polarization)
                    
        all_times_us = []
        for r_times in sim.solve_times:
            all_times_us.extend(r_times)
        mean_us = float(np.mean(all_times_us)) if all_times_us else 0.0
        max_ms = float(np.max(all_times_us) / 1000.0) if all_times_us else 0.0
        
        for i in range(n_robots):
            v_hist = np.linalg.norm(np.array(sim.v_log[i]), axis=1)
            w_hist = np.array(sim.w_log[i])
            if len(v_hist) >= 2:
                v_dot = np.diff(v_hist) / DT
                w_dot = np.diff(w_hist) / DT
                robot_jerks.append(np.mean(v_dot**2 + w_dot**2))
        mean_jerk = float(np.mean(robot_jerks)) if robot_jerks else 0.0
        
        feasibility_arr = np.array(feasibility_statuses)
        total_solves = feasibility_arr.size
        successful_solves = np.sum(feasibility_arr == 0)
        feasibility_rate = (successful_solves / total_solves) * 100.0 if total_solves > 0 else 0.0

        collisions_events = int(getattr(sim, "total_collisions", 0))

        sys.path = orig_sys_path

        return {
            "success": True,
            "folder": folder,
            "label": label,
            "n_robots": n_robots,
            "case_name": case_name,
            "seed": seed,
            "collision_history": collision_history,
            "collisions_events": collisions_events,
            "feasibility_rate": feasibility_rate,
            "mean_us": mean_us,
            "max_ms": max_ms,
            "mean_distance": float(np.mean(steady_state_distances)) if steady_state_distances else 0.4,
            "mean_polarization": float(np.mean(steady_state_polarizations)) if steady_state_polarizations else 0.0,
            "mean_jerk": mean_jerk
        }
        
    except Exception as e:
        sys.path = orig_sys_path
        return {"success": False, "error": str(e), "label": label, "n_robots": n_robots, "case_name": case_name}

def _plot_collision_vs_time_on_axis(ax, case_name, n, db, time_axis):
    """Draw raw collisions, rolling mean, and mean reference for all variants on one axis."""
    for _, label in SAFETY_LEVEL_FOLDERS:
        data = db[case_name][label].get(n)
        if not data:
            continue
        color = COLORS[label]
        raw_collisions = np.array(data["collision_history"])
        ax.plot(time_axis, raw_collisions, color=color, alpha=0.12, lw=0.8, zorder=1)

        w = 20
        if len(raw_collisions) >= w:
            padded = np.pad(raw_collisions, (w // 2, w // 2), mode="edge")
            rolling_avg = np.convolve(padded, np.ones(w) / w, mode="valid")[: len(raw_collisions)]
            ax.plot(
                time_axis, rolling_avg, label=label, color=color,
                alpha=0.95, lw=1.8, zorder=3,
            )
            overall_avg = np.mean(raw_collisions)
            ax.axhline(overall_avg, color=color, linestyle=":", alpha=0.35, lw=0.9, zorder=2)
        else:
            ax.plot(
                time_axis, raw_collisions, label=label, color=color,
                alpha=0.95, lw=1.8, zorder=3,
            )

    ax.set_xlim(0, SIM_TIME_SEC)
    ax.set_ylim(bottom=0)


def _combined_collision_figsize(n_cols: int, n_rows: int) -> tuple[float, float]:
    """Panel size scales down when many swarm-size columns are requested."""
    n_cols = max(n_cols, 1)
    n_rows = max(n_rows, 1)
    panel_w = max(2.8, min(4.2, 18.0 / n_cols))
    panel_h = 3.4
    return panel_w * n_cols, panel_h * n_rows + 1.35


def _combined_grid_panel_title(row_idx, col_idx, n):
    """Column header for the top row ($N_b$); empty for other rows."""
    if row_idx == 0:
        return f"$N_b = {n}$"
    return ""


def _generate_combined_collision_vs_time(db, time_axis, is_final=False):
    """
    Combined collision-vs-time figure.

    Layout is always ``len(CASES)`` rows × ``len(N_LIST)`` columns:
      - each row  = one benchmark case (phototaxis, orbital, …)
      - each column = one swarm size from ``N_LIST``
    """
    if not CASES or not N_LIST:
        return

    n_cols = len(N_LIST)
    n_rows = len(CASES)
    swarm_sizes = list(N_LIST)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=_combined_collision_figsize(n_cols, n_rows),
        facecolor="white",
        squeeze=False,
    )

    left_margin = max(0.14, 0.08 + 0.008 * n_cols)

    for row_idx, case in enumerate(CASES):
        case_name = case["name"]
        for col_idx, n in enumerate(swarm_sizes):
            ax = axes[row_idx, col_idx]
            _plot_collision_vs_time_on_axis(ax, case_name, n, db, time_axis)
            apply_premium_plot_style(
                ax,
                _combined_grid_panel_title(row_idx, col_idx, n),
                "Time (s)" if row_idx == n_rows - 1 else "",
                "Number of Active Collision" if col_idx == 0 else "",
                show_legend=False,
            )
            if col_idx == 0:
                add_subfigure_label(ax, row_idx)

    fig.suptitle(
        "Active Collision Count over Time",
        fontsize=FONTSIZE_SUPTITLE,
        fontweight="bold",
        y=0.995,
        color="#111111",
    )
    fig.tight_layout(rect=[left_margin, 0.07, 1, 0.96])
    add_figure_legend(fig, axes[0, 0], y=0.01, axes_grid=axes, single_row=True)
    save_publication_figure(
        fig,
        os.path.join(FIG_DIR, "combined_collision_vs_time"),
        is_final=is_final,
        log_label=f"{n_rows}×{n_cols} cases×N",
    )
    plt.close(fig)


def _density_xlim():
    return min(N_LIST) - 5, max(N_LIST) + 5


def _plot_feasibility_on_axis(ax, case_name, db):
    for _, label in SAFETY_LEVEL_FOLDERS:
        valid_n = [n for n in N_LIST if n in db[case_name][label]]
        if valid_n:
            rates = [db[case_name][label][n]["feasibility_rate"] for n in valid_n]
            ax.plot(valid_n, rates, marker="o", label=label, color=COLORS[label], lw=2.0, markersize=6)


def _plot_computation_on_axes(ax_mean, ax_peak, case_name, db):
    for _, label in SAFETY_LEVEL_FOLDERS:
        valid_n = [n for n in N_LIST if n in db[case_name][label]]
        if valid_n:
            means = [db[case_name][label][n]["mean_us"] for n in valid_n]
            peaks = [db[case_name][label][n]["max_ms"] for n in valid_n]
            ax_mean.plot(valid_n, means, marker="o", label=label, color=COLORS[label], lw=2.0, markersize=5)
            ax_peak.plot(
                valid_n, peaks, marker="^", linestyle="--", label=label,
                color=COLORS[label], lw=1.8, markersize=5,
            )


def _plot_target_distance_on_axis(ax, case_name, db):
    for _, label in SAFETY_LEVEL_FOLDERS:
        valid_n = [n for n in N_LIST if n in db[case_name][label]]
        if valid_n:
            dists = [db[case_name][label][n]["mean_distance"] for n in valid_n]
            ax.plot(valid_n, dists, marker="o", label=label, color=COLORS[label], lw=2.0, markersize=6)


def _plot_polarization_on_axis(ax, case_name, db):
    for _, label in SAFETY_LEVEL_FOLDERS:
        valid_n = [n for n in N_LIST if n in db[case_name][label]]
        if valid_n:
            pols = [db[case_name][label][n]["mean_polarization"] for n in valid_n]
            ax.plot(valid_n, pols, marker="^", label=label, color=COLORS[label], lw=2.0, markersize=6)


def _plot_jerk_on_axis(ax, case_name, db):
    for _, label in SAFETY_LEVEL_FOLDERS:
        valid_n = [n for n in N_LIST if n in db[case_name][label]]
        if valid_n:
            jerks = [db[case_name][label][n]["mean_jerk"] for n in valid_n]
            ax.plot(valid_n, jerks, marker="x", label=label, color=COLORS[label], lw=2.0, markersize=6)


def _combined_row_figsize(n_cols=None):
    n_cols = n_cols or len(CASES)
    return 7.0 * max(n_cols, 1), 4.6


def _generate_combined_single_row(
    db,
    plot_fn,
    *,
    suptitle,
    ylabel,
    basename,
    is_final=False,
    ylim=None,
    is_log=False,
    log_label="2 cases",
):
    """One row x len(CASES) columns: (a) phototaxis, (b) orbital, etc."""
    if not CASES:
        return

    n_cols = len(CASES)
    fig, axes = plt.subplots(1, n_cols, figsize=_combined_row_figsize(n_cols), facecolor="white", squeeze=False)

    for col_idx, case in enumerate(CASES):
        ax = axes[0, col_idx]
        plot_fn(ax, case["name"], db)
        apply_premium_plot_style(
            ax, "", "Number of Robots ($N_b$)", ylabel,
            is_log=is_log, show_legend=False,
        )
        ax.set_xlim(*_density_xlim())
        if ylim is not None:
            ax.set_ylim(*ylim)
        add_subfigure_label(ax, col_idx, x=-0.10)

    fig.suptitle(
        suptitle,
        fontsize=FONTSIZE_SUPTITLE,
        fontweight="bold",
        y=0.98,
        color="#111111",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    add_figure_legend(fig, axes[0, 0], y=0.02, axes_grid=axes)
    save_publication_figure(
        fig,
        os.path.join(FIG_DIR, basename),
        is_final=is_final,
        log_label=log_label,
    )
    plt.close(fig)


def _generate_combined_computation(db, is_final=False):
    """Two rows (cases) × two columns (mean latency, peak latency)."""
    if not CASES:
        return

    n_rows = len(CASES)
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(14.0, 4.6 * n_rows + 0.35),
        facecolor="white",
        squeeze=False,
    )

    for row_idx, case in enumerate(CASES):
        _plot_computation_on_axes(axes[row_idx, 0], axes[row_idx, 1], case["name"], db)
        apply_premium_plot_style(
            axes[row_idx, 0],
            "Average MPC Calculation Latency",
            "Number of Robots ($N_b$)" if row_idx == n_rows - 1 else "",
            "Mean Solve Time per Step (μs)",
            is_log=True, show_legend=False,
        )
        apply_premium_plot_style(
            axes[row_idx, 1],
            "Worst-Case Peak Solver Latency (Jitter)",
            "Number of Robots ($N_b$)" if row_idx == n_rows - 1 else "",
            "Peak Solve Time (ms)" if row_idx == 0 else "",
            is_log=True, show_legend=False,
        )
        for col_idx in range(2):
            axes[row_idx, col_idx].set_xlim(*_density_xlim())
        add_subfigure_label(axes[row_idx, 0], row_idx)

    fig.suptitle(
        "OCP Distributed Solver Computational Performance & Latency vs. Swarm Size",
        fontsize=FONTSIZE_SUPTITLE,
        fontweight="bold",
        y=0.995,
        color="#111111",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    add_figure_legend(fig, axes[0, 0], y=0.02, axes_grid=axes)
    save_publication_figure(
        fig,
        os.path.join(FIG_DIR, "combined_computation_vs_density"),
        is_final=is_final,
        log_label=f"{n_rows}×2 cases×latency",
    )
    plt.close(fig)


def _generate_all_combined_plots(db, time_axis, is_final=False):
    """Generate every combined phototaxis + orbital figure."""
    _generate_combined_collision_vs_time(db, time_axis, is_final=is_final)
    _generate_combined_single_row(
        db, _plot_feasibility_on_axis,
        suptitle="OCP Solver Feasibility Success Rate vs. Swarm Size",
        ylabel="Overall Solver Success Rate (%)",
        basename="combined_feasibility_vs_density",
        ylim=(0, 105),
        is_final=is_final,
    )
    _generate_combined_computation(db, is_final=is_final)
    _generate_combined_single_row(
        db, _plot_target_distance_on_axis,
        suptitle="Steady-State Swarm Distance to Target vs. Swarm Size",
        ylabel="Average Target Distance (m)",
        basename="combined_target_distance_vs_density",
        ylim=(0.0, 0.40),
        is_final=is_final,
    )
    _generate_combined_single_row(
        db, _plot_polarization_on_axis,
        suptitle="Steady-State Swarm Polarization vs. Swarm Size",
        ylabel=r"Average Swarm Polarization ($\Phi$)",
        basename="combined_polarization_vs_density",
        ylim=(0.0, 1.05),
        is_final=is_final,
    )
    _generate_combined_single_row(
        db, _plot_jerk_on_axis,
        suptitle="Average Swarm Control Command Jerk vs. Swarm Size",
        ylabel=r"Average Control Jerk ($(\mathrm{m/s^2})^2 + (\mathrm{rad/s^2})^2$)",
        basename="combined_jerk_vs_density",
        is_log=True,
        is_final=is_final,
    )


SCALAR_METRIC_KEYS = [
    "feasibility_rate", "mean_us", "max_ms",
    "mean_distance", "mean_polarization", "mean_jerk",
]


def aggregate_results(results):
    """Collapse per-seed raw results to one record per (case, label, n) holding the
    seed mean under each canonical metric key (so the existing plotters work
    unchanged) plus ``*_ci95`` keys. Two collision summaries are produced:
      - ``collisions_events_mean/std/ci95`` - distinct contacts (de-duplicated
        rising-edge ``sim.total_collisions``); this is what the figure/table use.
      - ``collisions_mean/std/ci95`` - legacy summed per-step overlap-pair-frames.
    95% CI = 1.96 * std(ddof=1) / sqrt(n)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        if r.get("success"):
            groups[(r["case_name"], r["label"], r["n_robots"])].append(r)

    def _stats(vals):
        a = np.asarray(vals, dtype=float)
        n = a.size
        mean = float(a.mean()) if n else 0.0
        std = float(a.std(ddof=1)) if n > 1 else 0.0
        ci95 = float(1.96 * std / np.sqrt(n)) if n > 1 else 0.0
        return mean, std, ci95

    agg = []
    for (case_name, label, n), rs in groups.items():
        rec = {"success": True, "case_name": case_name, "label": label,
               "n_robots": n, "n_seeds": len(rs)}
        for k in SCALAR_METRIC_KEYS:
            mean, std, ci95 = _stats([r[k] for r in rs])
            rec[k] = mean
            rec[k + "_std"] = std
            rec[k + "_ci95"] = ci95
        e_mean, e_std, e_ci95 = _stats(
            [float(r.get("collisions_events", 0)) for r in rs]
        )
        rec["collisions_events_mean"] = e_mean
        rec["collisions_events_std"] = e_std
        rec["collisions_events_ci95"] = e_ci95
        c_mean, c_std, c_ci95 = _stats([float(np.sum(r["collision_history"])) for r in rs])
        rec["collisions_mean"] = c_mean
        rec["collisions_std"] = c_std
        rec["collisions_ci95"] = c_ci95
        rep = min(rs, key=lambda r: r.get("seed", 0))
        rec["collision_history"] = rep["collision_history"]
        agg.append(rec)
    return agg


def generate_plots(results, is_final=False):
    """Regenerate and save all combined benchmark figures from the results so far."""
    if not results:
        return

    db = {}
    for case in CASES:
        db[case["name"]] = {}
        for _, label in SAFETY_LEVEL_FOLDERS:
            db[case["name"]][label] = {}
            
    for res in results:
        if res["success"]:
            db[res["case_name"]][res["label"]][res["n_robots"]] = res
            
    time_axis = np.arange(STEPS) * DT

    for case in CASES:
        case_name = case["name"]
        case_display = case["display_name"]
        suffix = case["suffix"]

        if is_final and SAVE_CSV:
            import csv
            
            collision_csv_path = os.path.join(CSV_DIR, f"{suffix}_collision_history.csv")
            try:
                with open(collision_csv_path, 'w', newline='') as f_csv:
                    writer = csv.writer(f_csv)
                    headers = ["time_step", "time_sec"]
                    cols_data = []
                    for _, label in SAFETY_LEVEL_FOLDERS:
                        for n in N_LIST:
                            data = db[case_name][label].get(n)
                            if data:
                                headers.append(f"{label}_N_{n}")
                                cols_data.append(data["collision_history"])
                    writer.writerow(headers)
                    
                    for step_idx in range(STEPS):
                        row = [step_idx, f"{step_idx * DT:.2f}"]
                        for col in cols_data:
                            if step_idx < len(col):
                                row.append(col[step_idx])
                            else:
                                row.append("")
                        writer.writerow(row)
                print(f"  ├── Saved CSV: {os.path.basename(collision_csv_path)}")
            except Exception as e_csv:
                print(f"  [WARN] Failed to save collision CSV: {e_csv}")
            
            metrics_csv_path = os.path.join(CSV_DIR, f"{suffix}_metrics_vs_density.csv")
            try:
                with open(metrics_csv_path, 'w', newline='') as f_csv:
                    writer = csv.writer(f_csv)
                    headers = [
                        "safety_level", "n_robots", "n_seeds",
                        "feasibility_rate_pct", "feasibility_rate_pct_ci95",
                        "feasibility_ci95",
                        "mean_solve_time_us", "mean_solve_time_us_ci95",
                        "max_solve_time_ms", "max_solve_time_ms_ci95",
                        "mean_target_distance_m", "mean_target_distance_m_ci95",
                        "mean_polarization", "mean_polarization_ci95",
                        "mean_jerk", "mean_jerk_ci95",
                        "collisions_events_mean", "collisions_events_ci95",
                        "collisions_total_mean", "collisions_total_ci95",
                    ]
                    writer.writerow(headers)
                    for _, label in SAFETY_LEVEL_FOLDERS:
                        for n in N_LIST:
                            data = db[case_name][label].get(n)
                            if data:
                                g = data.get
                                writer.writerow([
                                    label, n, g("n_seeds", 1),
                                    f"{data['feasibility_rate']:.2f}", f"{g('feasibility_rate_ci95', 0.0):.2f}",
                                    f"{g('feasibility_rate_ci95', 0.0):.2f}",
                                    f"{data['mean_us']:.2f}", f"{g('mean_us_ci95', 0.0):.2f}",
                                    f"{data['max_ms']:.4f}", f"{g('max_ms_ci95', 0.0):.4f}",
                                    f"{data['mean_distance']:.4f}", f"{g('mean_distance_ci95', 0.0):.4f}",
                                    f"{data['mean_polarization']:.4f}", f"{g('mean_polarization_ci95', 0.0):.4f}",
                                    f"{data['mean_jerk']:.4f}", f"{g('mean_jerk_ci95', 0.0):.4f}",
                                    f"{g('collisions_events_mean', 0.0):.2f}", f"{g('collisions_events_ci95', 0.0):.2f}",
                                    f"{g('collisions_mean', 0.0):.2f}", f"{g('collisions_ci95', 0.0):.2f}",
                                ])
                print(f"  ├── Saved CSV: {os.path.basename(metrics_csv_path)}")
            except Exception as e_csv:
                print(f"  [WARN] Failed to save metrics CSV: {e_csv}")

    if SAVE_PLOTS and CASES and N_LIST:
        if is_final:
            print("\nFormatting combined benchmark plots ...")
        _generate_all_combined_plots(db, time_axis, is_final=is_final)

def run_parallel_benchmark(enable_obstacles=False, workers_override=None):
    print("=" * 80)
    print("  STEP 2: Running swarm simulation sweeps in parallel")
    print("=" * 80)

    cores_available = multiprocessing.cpu_count()
    if workers_override is not None and workers_override > 0:
        workers = workers_override
    else:
        workers = max(1, cores_available - 4)
    print(f"  Detected CPU cores:  {cores_available}")
    print(f"  Parallel workers:    {workers}")
    print(f"  Arena obstacles:     {'ON' if enable_obstacles else 'OFF'}")
    print("-" * 80)
    
    all_tasks = []
    for case in CASES:
        case_name = case["name"]
        for folder, label in SAFETY_LEVEL_FOLDERS:
            for n in N_LIST:
                for seed in SEEDS:
                    all_tasks.append((folder, label, n, case_name, enable_obstacles, seed))

    ckpt_dir = os.path.join(RESULTS_DIR, "_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    def _san(s):
        return re.sub(r"[^A-Za-z0-9]+", "_", str(s))

    def _ckpt_path(case_name, label, n, seed):
        return os.path.join(ckpt_dir, f"{_san(case_name)}__{_san(label)}__N{n}__s{seed}.pkl")

    results = []
    done = set()
    for fn in os.listdir(ckpt_dir):
        if not fn.endswith(".pkl"):
            continue
        fp = os.path.join(ckpt_dir, fn)
        try:
            with open(fp, "rb") as fh:
                res = pickle.load(fh)
            key = (res["case_name"], res["label"], res["n_robots"], res.get("seed", 0))
            if key in {(c, l, n, s) for (_, l, n, c, _, s) in all_tasks}:
                results.append(res)
                done.add(key)
        except Exception:
            try:
                os.remove(fp)
            except OSError:
                pass

    tasks = [
        t for t in all_tasks
        if (t[3], t[1], t[2], t[5]) not in done
    ]

    total_tasks = len(all_tasks)
    pending = len(tasks)
    print(f"  Total sweep: {total_tasks} tasks ({len(SEEDS)} seeds). "
          f"Resuming: {len(done)} already done, {pending} pending.")

    start_t = time.time()

    with multiprocessing.Pool(processes=workers) as pool:
        for idx, res in enumerate(pool.imap_unordered(sim_worker_task, tasks)):
            if res["success"]:
                results.append(res)
                key = (res["case_name"], res["label"], res["n_robots"], res.get("seed", 0))
                done.add(key)
                fp = _ckpt_path(*key)
                try:
                    with open(fp + ".tmp", "wb") as fh:
                        pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(fp + ".tmp", fp)
                except Exception as e_ck:
                    print(f"  [WARN] checkpoint write failed for {key}: {e_ck}")
                ndone = len(done)
                if ndone % len(SEEDS) == 0 or idx + 1 == pending:
                    print(f"  [OK {ndone:3d}/{total_tasks:3d}] {res['label']:18s} | N={res['n_robots']:3d} | seed={res.get('seed', 0)} | Case: {res['case_name']:28s}")
            else:
                print(f"  [FAIL {idx+1:3d}/{pending:3d}] {res.get('label','?'):18s} | N={res.get('n_robots',0):3d} | Case: {res.get('case_name','?'):28s} -> {res.get('error', 'unknown error')}")

    elapsed = time.time() - start_t
    print("-" * 80)
    print(f"Parallel sweeps completed in {elapsed:.1f} seconds.")
    print("=" * 80 + "\n")
    
    aggregated = aggregate_results(results)
    print(f"  Aggregated {len(results)} runs into {len(aggregated)} (case x variant x N) records over {len(SEEDS)} seeds.")
    if SAVE_PLOTS:
        print("STEP 3: Generating final publication figures...")
    elif SAVE_CSV:
        print("STEP 3: Saving final benchmark metrics as CSV...")
    generate_plots(aggregated, is_final=True)

    print("\n" + "=" * 80)
    if SAVE_PLOTS and SAVE_CSV:
        print("  Done: solvers compiled, sweeps run, CSVs and plots saved.")
    elif SAVE_PLOTS:
        print("  Done: solvers compiled, sweeps run, plots saved.")
    elif SAVE_CSV:
        print("  Done: solvers compiled, sweeps run, CSVs saved.")
    else:
        print("  Done: solvers compiled and sweeps completed.")
    print(f"  Results Directory: {RESULTS_DIR}")
    print("=" * 80 + "\n")

def main():
    global SAVE_CSV, SAVE_PLOTS, SEEDS

    import argparse
    parser = argparse.ArgumentParser(description="High-Performance Parallel Workflow Runner for DMPC Swarm Safety Benchmarks.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--csv-only", action="store_true", help="Save CSV files only (disable plots/diagrams).")
    group.add_argument("--plots-only", "--diagrams-only", action="store_true", help="Generate plots/diagrams only (disable CSV).")
    
    parser.add_argument("--no-csv", action="store_true", help="Disable saving CSV files.")
    parser.add_argument("--no-plots", "--no-diagrams", action="store_true", help="Disable generating plots/diagrams.")
    parser.add_argument(
        "--obstacles",
        action="store_true",
        help="Enable static arena obstacles (moving obstacles disabled in sim config).",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip Acados solver recompilation (solvers already built).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Override the number of replication seeds (default: len(SEEDS) from config).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override the number of parallel worker processes (default: CPU cores - 4).",
    )

    args = parser.parse_args()

    if args.seeds is not None:
        SEEDS = list(range(args.seeds))
    
    if args.csv_only:
        SAVE_CSV = True
        SAVE_PLOTS = False
    elif args.plots_only:
        SAVE_CSV = False
        SAVE_PLOTS = True
    else:
        if args.no_csv:
            SAVE_CSV = False
        if args.no_plots:
            SAVE_PLOTS = False

    configure_results_paths(enable_obstacles=args.obstacles)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if SAVE_CSV:
        os.makedirs(CSV_DIR, exist_ok=True)

    if args.skip_compile:
        print("Skipping solver compilation (--skip-compile).")
    else:
        compile_solvers()

    run_parallel_benchmark(enable_obstacles=args.obstacles, workers_override=args.workers)

if __name__ == "__main__":
    main()
