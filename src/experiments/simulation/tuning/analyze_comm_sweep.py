#!/usr/bin/env python3
"""Summarises the communication-radius sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "results_sweep" / "comm_k_r_sweep.csv"
OUT_PATH = SCRIPT_DIR / "results_sweep" / "comm_k_r_ranked.csv"
DEFAULT_VARIANT = "Hybrid Slacked BF+Dt-CBF"


def _load() -> list[dict]:
    with CSV_PATH.open() as f:
        return list(csv.DictReader(f))


def _aggregate(rows: list[dict], variant: str) -> list[dict]:
    filt = [r for r in rows if r["label"] == variant and r["is_operating_max"] == "1"]
    groups: dict[tuple, list[dict]] = {}
    for r in filt:
        key = (int(r["k"]), float(r["comm_radius_m"]))
        groups.setdefault(key, []).append(r)
    out = []
    for (k, r), grp in sorted(groups.items()):
        out.append({
            "k": k,
            "comm_radius_m": r,
            "n_operating": int(grp[0]["n_operating"]),
            "mean_collision_frames": sum(int(x["collision_frames"]) for x in grp) / len(grp),
            "mean_feasibility_pct": sum(float(x["feasibility_rate_pct"]) for x in grp) / len(grp),
            "mean_solve_time_us": sum(float(x["mean_solve_time_us"]) for x in grp) / len(grp),
            "mean_mpc_neighbors": sum(float(x["mean_mpc_neighbors"]) for x in grp) / len(grp),
            "mean_slot_fill_pct": sum(float(x["slot_fill_pct"]) for x in grp) / len(grp),
        })
    return out


def _norm(vals: list[float], hi_better: bool) -> list[float]:
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [1.0] * len(vals)
    n = [(v - lo) / (hi - lo) for v in vals]
    return n if hi_better else [1.0 - x for x in n]


def _score(rows: list[dict]) -> list[dict]:
    if not rows:
        return rows
    coll = _norm([r["mean_collision_frames"] for r in rows], False)
    feas = _norm([r["mean_feasibility_pct"] for r in rows], True)
    time_ = _norm([r["mean_solve_time_us"] for r in rows], False)
    fill = _norm([r["mean_slot_fill_pct"] for r in rows], True)
    k_n = _norm([r["k"] for r in rows], False)
    r_n = _norm([r["comm_radius_m"] for r in rows], False)
    for i, r in enumerate(rows):
        r["score"] = round(
            0.25 * coll[i] + 0.20 * feas[i] + 0.15 * time_[i]
            + 0.15 * fill[i] + 0.15 * k_n[i] + 0.10 * r_n[i],
            4,
        )
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    args = parser.parse_args()

    rows = _load()
    agg = _aggregate(rows, args.variant)
    viable = [r for r in agg if r["mean_slot_fill_pct"] >= 80.0]
    agg = _score(viable if viable else agg)
    if not agg:
        print("No data")
        return

    with OUT_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(agg)

    print(f"=== {args.variant} @ N_operating (ranked) ===")
    print(f"{'K':>3} {'R':>6} {'fill%':>6} {'coll':>8} {'feas%':>6} {'score':>6}")
    for r in agg[:12]:
        print(
            f"{r['k']:3d} {r['comm_radius_m']:6.2f} {r['mean_slot_fill_pct']:6.1f} "
            f"{r['mean_collision_frames']:8.0f} {r['mean_feasibility_pct']:6.1f} {r['score']:6.3f}"
        )
    best = agg[0]
    print(f"\nRecommended: K={best['k']}, R={best['comm_radius_m']:.2f} m")


if __name__ == "__main__":
    main()
