#!/usr/bin/env python3
"""Loads matrix runs and computes their steady-state averages."""
import os, csv, glob, argparse
import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("MATRIX_OUT_DIR",
                     os.path.abspath(os.path.join(HERE, "..", "results", "csv", "matrix")))
DT, D_SAFE_CM = 0.1, 7.0
SS_TOL, SS_TAIL, SS_SMOOTH_S = 0.05, 0.2, 15.0
SS_FLOOR_S = 50.0
STEPS = 9000

VARIANTS = ["01", "04", "05"]
BEHAVIORS = [("phototaxis", "Phototaxis"), ("phototaxis_orbital", "Orbital"),
             ("phototaxis_orbital_contracting", "Orbital (poc)")]
NS = [10, 25, 50, 75, 100, 125, 137]
OBS = [0, 2, 12]
COLS = ["t_sec", "min_dist_m", "avg_min_sep_m", "mean_dist_m", "coherence",
        "local_polarization", "comm_neighbor_mean_dist_m", "local_nematic",
        "global_nematic", "cum_collisions", "active_collisions", "safety_violations",
        "mean_path_len_m"]

PANELS = [
    ("avg_min_sep_m", "Mean nearest-nbr dist (cm)", 100.0, None),
    ("mean_dist_m", "Mean pairwise dist (cm)", 100.0, None),
    ("comm_neighbor_mean_dist_m", "Mean comm-nbr dist (cm)", 100.0, None),
    ("mean_path_len_m", "Mean path length (m)", 1.0, None),
    ("__active__", "Active collisions (overlapping pairs)", 1.0, None),
    ("coherence", r"Global polarization $\Phi$", 1.0, (0, 1.02)),
    ("local_polarization", r"Local polarization $\Phi_{\rm loc}$", 1.0, (0, 1.02)),
    ("local_nematic", r"Local nematic order $S_{\rm loc}$", 1.0, (0, 1.02)),
    ("global_nematic", r"Global nematic order $S$", 1.0, (0, 1.02)),
    ("__coll__", "Cumulative collisions", 1.0, None),
]


def _f(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan


def load_seeds(v, beh, n, o):
    runs = []
    for p in sorted(glob.glob(os.path.join(RES, f"v{v}_{beh}_N{n}_obs{o}_reacon_seed*_steps{STEPS}.csv"))):
        rows = list(csv.DictReader(open(p, newline="")))
        if rows:
            runs.append({k: np.array([_f(r[k]) for r in rows]) for k in COLS})
    return runs


def settling_index(y):
    y = np.asarray(y, float)
    n = len(y)
    if n < 10:
        return 0
    w = max(1, int(round(SS_SMOOTH_S / DT)))
    ys = uniform_filter1d(y, size=w, mode="nearest")
    ref = ys[int(n * (1 - SS_TAIL)):].mean()
    band = SS_TOL * abs(ref) if abs(ref) > 1e-9 else max(SS_TOL, 1e-3)
    idx = np.where(np.abs(ys - ref) > band)[0]
    return min(int(idx[-1] + 1) if idx.size else 0, n - 1)


def group(runs, key):
    L = min(len(r["t_sec"]) for r in runs)
    M = np.stack([r[key][:L] for r in runs])
    return M[:, :L], L


def facet_figure(v, beh, beh_label, o):
    plt.rcParams.update({"font.family": "serif", "font.size": 9.5, "axes.grid": True,
                         "grid.linestyle": ":", "grid.alpha": 0.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 5, figsize=(21, 7.4), sharex=True)
    axes = axes.ravel()
    colors = cm.viridis(np.linspace(0.05, 0.9, len(NS)))
    any_data = False
    for ax, (key, label, scale, ylim) in zip(axes, PANELS):
        for ni, n in enumerate(NS):
            runs = load_seeds(v, beh, n, o)
            if not runs:
                continue
            any_data = True
            ck = {"__coll__": "cum_collisions", "__active__": "active_collisions"}.get(key, key)
            M, L = group(runs, ck)
            t = runs[0]["t_sec"][:L]
            mean = np.nanmean(M, axis=0) * scale
            ax.plot(t, mean, color=colors[ni], lw=1.6, label=f"N={n}")
        if key == "avg_min_sep_m":
            ax.axhline(D_SAFE_CM, ls="--", color="#C44E52", lw=0.9)
        if key == "__coll__":
            ax.set_yscale("symlog", linthresh=1)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(label, fontsize=9.5)
        ax.set_axisbelow(True)
    for ax in axes[5:]:
        ax.set_xlabel("Time (s)")
    if not any_data:
        plt.close(fig)
        return None
    axes[0].legend(fontsize=8, loc="best", frameon=True)
    obs_label = {0: "free space", 2: "2 obstacles", 12: "12 obstacles"}[o]
    fig.suptitle(f"Variant {v} - {beh_label} - {obs_label} - reactive ON  "
                 f"(mean over seeds; N in {{{','.join(str(n) for n in NS)}}})", y=1.0, fontsize=12)
    fig.tight_layout()
    out = os.path.join(RES, f"matrix_v{v}_{beh}_obs{o}.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def build_summary():
    rows = []
    for v in VARIANTS:
        for beh, _ in BEHAVIORS:
            for n in NS:
                for o in OBS:
                    runs = load_seeds(v, beh, n, o)
                    if not runs:
                        continue
                    ns = len(runs)
                    md, L = group(runs, "mean_dist_m")
                    lp, _ = group(runs, "local_polarization")
                    d_ss = max(settling_index(np.nanmean(md, 0)),
                               settling_index(np.nanmean(lp, 0)),
                               int(round(SS_FLOOR_S / DT)))
                    d_ss = min(d_ss, L - 1)
                    t = runs[0]["t_sec"][:L]
                    cum, _ = group(runs, "cum_collisions")
                    path, _ = group(runs, "mean_path_len_m")
                    def wmean(key):
                        M, _ = group(runs, key)
                        return float(np.nanmean(M[:, d_ss:]))
                    rows.append({
                        "variant": v, "behavior": beh, "N": n, "obstacles": o, "seeds": ns,
                        "t_ss_s": round(float(t[d_ss]), 1),
                        "coll_overall": round(float(cum[:, -1].mean()), 1),
                        "coll_window": round(float((cum[:, -1] - cum[:, d_ss]).mean()), 1),
                        "path_len_total_m": round(float(path[:, -1].mean()), 3),
                        "min_dist_cm_win": round(wmean("min_dist_m") * 100, 2),
                        "avg_min_sep_cm_win": round(wmean("avg_min_sep_m") * 100, 2),
                        "mean_dist_cm_win": round(wmean("mean_dist_m") * 100, 2),
                        "comm_dist_cm_win": round(wmean("comm_neighbor_mean_dist_m") * 100, 2),
                        "Phi_win": round(wmean("coherence"), 3),
                        "Phi_loc_win": round(wmean("local_polarization"), 3),
                        "nematic_loc_win": round(wmean("local_nematic"), 3),
                        "nematic_glob_win": round(wmean("global_nematic"), 3),
                        "safety_viol_win": round(wmean("safety_violations"), 2),
                    })
    out = os.path.join(RES, "steady_state_matrix_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows, out


def method_compare(rows, metric, ylabel, logy=False):
    """01 vs 08 vs N, faceted behavior x obstacle."""
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, axes = plt.subplots(len(BEHAVIORS), len(OBS), figsize=(13, 7), sharex=True)
    mcol = {"01": "#C44E52", "04": "#55A868", "05": "#4C72B0"}
    mlab = {"01": "01 hard BF", "07": "07 hybrid (no look-ahead)",
            "05": "05 slacked Dt-CBF with look-ahead"}
    for bi, (beh, blab) in enumerate(BEHAVIORS):
        for oi, o in enumerate(OBS):
            ax = axes[bi][oi]
            for v in VARIANTS:
                ys = []
                for n in NS:
                    r = [x for x in rows if x["variant"] == v and x["behavior"] == beh
                         and x["N"] == n and x["obstacles"] == o]
                    ys.append(r[0][metric] if r else np.nan)
                ax.plot(NS, ys, "o-", color=mcol[v], lw=1.8, ms=5, label=mlab.get(v, f"variant {v}"))
            if logy:
                ax.set_yscale("symlog", linthresh=1)
            ax.grid(True, ls=":", alpha=0.5)
            if bi == 0:
                ax.set_title({0: "free", 2: "2 obs", 12: "12 obs"}[o], fontsize=10)
            if oi == 0:
                ax.set_ylabel(f"{blab}\n{ylabel}", fontsize=9)
            if bi == len(BEHAVIORS) - 1:
                ax.set_xlabel("N robots")
    axes[0][0].legend(fontsize=9, loc="best", frameon=True)
    fig.suptitle(f"Variant 01 (hard-BF) vs 07 (hybrid, no look-ahead) vs 08 (hybrid, look-ahead): "
                 f"{ylabel} vs swarm size", y=1.0, fontsize=12.5)
    fig.tight_layout()
    out = os.path.join(RES, f"compare_01v05_{metric}.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    figs = []
    for v in VARIANTS:
        for beh, blab in BEHAVIORS:
            for o in OBS:
                out = facet_figure(v, beh, blab, o)
                if out:
                    figs.append(out)
    print(f"facet figures: {len(figs)}")
    rows, csv_out = build_summary()
    print(f"summary rows: {len(rows)} -> {csv_out}")
    for metric, ylab, logy in [("coll_overall", "Collisions (overall)", True),
                               ("nematic_loc_win", "Local nematic order S", False),
                               ("nematic_glob_win", "Global nematic order S", False),
                               ("comm_dist_cm_win", "Comm-nbr dist (cm)", False)]:
        print("compare:", method_compare(rows, metric, ylab, logy))


if __name__ == "__main__":
    main()
