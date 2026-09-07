#!/usr/bin/env python3
"""Works out which benchmark task ids cover a chosen subset of the grid."""
import argparse

FULL_NS = [10, 25, 50, 75, 100, 125, 137]
N_VARIANTS = 6
BENCH_SEEDS = 10

ap = argparse.ArgumentParser()
ap.add_argument("kind", choices=["bench", "distance"])
ap.add_argument("--ns", default=" ".join(map(str, FULL_NS)))
ap.add_argument("--seeds", type=int, default=10)
a = ap.parse_args()

want = [int(x) for x in a.ns.split()]
bad = [n for n in want if n not in FULL_NS]
if bad:
    raise SystemExit(f"densities {bad} are not on the grid {FULL_NS}")
idx = [FULL_NS.index(n) for n in want]
stride = BENCH_SEEDS if a.kind == "bench" else a.seeds

for v in range(N_VARIANTS):
    for ni in idx:
        for s in range(a.seeds):
            print((v * len(FULL_NS) + ni) * stride + s)
