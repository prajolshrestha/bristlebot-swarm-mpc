#!/usr/bin/env python3
"""Summarises the safety-distance sweep."""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results_sweep"
SUMMARY_CSV = RESULTS_ROOT / "sweep_ranked_summary.csv"

DEFAULT_VARIANT = "Hybrid Slacked BF+Dt-CBF"


def _load_all_rows() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(RESULTS_ROOT.glob("dsafe_*cm.csv")):
        with path.open() as f:
            rows.extend(csv.DictReader(f))
    return rows


def _aggregate_operating(rows: list[dict], variant: str) -> list[dict]:
    """One row per (d_safe, headroom) at N_operating for the chosen variant."""
    filtered = [
        r for r in rows
        if r["safety_level"] == variant and r["is_operating_max"] == "1"
    ]
    groups: dict[tuple, list[dict]] = {}
    for r in filtered:
        key = (float(r["d_safe_cm"]), int(r["headroom"]))
        groups.setdefault(key, []).append(r)

    out: list[dict] = []
    for (d_safe, headroom), grp in sorted(groups.items()):
        n_op = int(grp[0]["n_operating"])
        n_geo = int(grp[0]["n_geo"])
        buf_occ = float(grp[0]["buffer_occ_pct"])
        collisions = sum(int(r["collision_frames"]) for r in grp) / len(grp)
        feasibility = sum(float(r["feasibility_rate_pct"]) for r in grp) / len(grp)
        solve_us = sum(float(r["mean_solve_time_us"]) for r in grp) / len(grp)
        comm = sum(float(r["mean_neighbors"]) for r in grp) / len(grp)
        out.append({
            "d_safe_cm": d_safe,
            "headroom": headroom,
            "n_geo": n_geo,
            "n_operating": n_op,
            "buffer_occ_pct": buf_occ,
            "mean_collision_frames": collisions,
            "mean_feasibility_pct": feasibility,
            "mean_infeasibility_pct": 100.0 - feasibility,
            "mean_solve_time_us": solve_us,
            "mean_neighbors": comm,
            "n_cases": len(grp),
        })
    return out


def _normalize(values: list[float], higher_better: bool) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    normed = [(v - lo) / (hi - lo) for v in values]
    return normed if higher_better else [1.0 - x for x in normed]


def _score(rows: list[dict]) -> list[dict]:
    if not rows:
        return rows
    occ = _normalize([r["n_operating"] for r in rows], True)
    coll = _normalize([r["mean_collision_frames"] for r in rows], False)
    feas = _normalize([r["mean_feasibility_pct"] for r in rows], True)
    time_ = _normalize([r["mean_solve_time_us"] for r in rows], False)
    comm = _normalize([r["mean_neighbors"] for r in rows], False)
    for i, r in enumerate(rows):
        r["score"] = round(
            0.30 * occ[i] + 0.25 * coll[i] + 0.20 * feas[i] + 0.15 * time_[i] + 0.10 * comm[i],
            4,
        )
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    args = parser.parse_args()

    rows = _load_all_rows()
    if not rows:
        print(f"No sweep CSVs found in {RESULTS_ROOT}")
        return

    agg = _aggregate_operating(rows, args.variant)
    ranked = _score(agg)

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ranked[0].keys()))
        w.writeheader()
        w.writerows(ranked)

    print(f"\n=== Ranked (d_safe, headroom) at N_operating: {args.variant} ===")
    print(f"{'Rank':>4} {'d_safe':>6} {'room':>4} {'N_op':>4} {'buf%':>5} "
          f"{'coll':>8} {'feas%':>6} {'solveµs':>8} {'nb':>4} {'score':>6}")
    print("-" * 72)
    for i, r in enumerate(ranked[:15], 1):
        print(
            f"{i:4d} {r['d_safe_cm']:6.0f} {r['headroom']:4d} {r['n_operating']:4d} "
            f"{r['buffer_occ_pct']:5.1f} {r['mean_collision_frames']:8.0f} "
            f"{r['mean_feasibility_pct']:6.1f} {r['mean_solve_time_us']:8.0f} "
            f"{r['mean_neighbors']:4.1f} {r['score']:6.3f}"
        )

    best = ranked[0]
    print(
        f"\nRecommended: D_safe={best['d_safe_cm']:.0f} cm, "
        f"headroom={best['headroom']}, N={best['n_operating']} "
        f"(score={best['score']:.3f})"
    )
    print(f"Full ranking → {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
