#!/usr/bin/env python3
"""Checks the regenerated benchmark against the numbers behind the published paper."""
import os
import sys
import csv
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.join(HERE, "results", "csv", "phototaxis_metrics_vs_density.csv")
PAPER = ("/home/hpc/iwi5/iwi5171h/different_dmpc/swarm_mpc/comparison/comparison_code/"
         "deterministic/results_bench_900s_tuned/csv_data/phototaxis_metrics_vs_density.csv")
KEY = "collisions_events_mean"
LEGACY = {"Hybrid": "Slack Dt-CBF", "Hybrid v2": "LA Slack Dt-CBF",
          "Dt-HOCBF": "Slack Dt-HOCBF"}


def load(path):
    """Grid point -> row, with the reference CSV's legacy variant names mapped."""
    with open(path, newline="") as fh:
        return {(LEGACY.get(r["safety_level"], r["safety_level"]), int(r["n_robots"])): r
                for r in csv.DictReader(fh)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-csv", default=NEW)
    ap.add_argument("--paper-csv", default=PAPER)
    ap.add_argument("--tol", type=float, default=5.0)
    a = ap.parse_args()

    for p in (a.new_csv, a.paper_csv):
        if not os.path.isfile(p):
            sys.exit("missing: " + p)

    new, paper = load(a.new_csv), load(a.paper_csv)
    labels = sorted({k[0] for k in paper} & {k[0] for k in new})
    ns = sorted({k[1] for k in paper} & {k[1] for k in new})

    print(f"{KEY}: regenerated vs paper, percent difference\n")
    print(" " * 18 + "".join(f"{'N=' + str(n):>10}" for n in ns))
    worst, bad = 0.0, []
    for lab in labels:
        row = f"  {lab:<16}"
        for n in ns:
            rn, rp = new.get((lab, n)), paper.get((lab, n))
            if rn is None or rp is None:
                row += "       n/a"
                continue
            b_, a_ = float(rn[KEY]), float(rp[KEY])
            if a_ == 0:
                row += "      same" if b_ == 0 else f"{b_:>10.0f}"
                continue
            pct = 100.0 * (b_ - a_) / a_
            worst = max(worst, abs(pct))
            if abs(pct) > a.tol:
                bad.append((lab, n, b_, a_, pct))
            row += f"{pct:>+10.1f}"
        print(row)

    print(f"\n  largest deviation: {worst:.1f}%  (tolerance {a.tol:.1f}%)")
    if bad:
        print(f"  {len(bad)} grid point(s) beyond tolerance:")
        for lab, n, b_, a_, pct in bad:
            print(f"    {lab} N={n} regenerated={b_:.0f}  paper={a_:.0f}  ({pct:+.1f}%)")
        sys.exit(1)
    print("  every grid point within tolerance: the port reproduces the paper.")


if __name__ == "__main__":
    main()
