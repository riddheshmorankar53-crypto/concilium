"""Stage 1b analysis exactly as pre-registered (damp_stage1b_prereg.md): python results/damp_stage1b_analysis.py"""
import os
from math import comb

import numpy as np

rows = [l.strip().split("|") for l in open(os.path.join(os.path.dirname(__file__), "damp_stage1b.log"))]
runs = {}  # (seed, draw) -> variant -> (acc, honest rejected, skipped)
for seed, draw, _, var, acc, _, hon, _, skip in rows:
    runs.setdefault((seed, draw), {})[var.split()[0]] = (float(acc), float(hon), int(skip))
pairs = [r for r in runs.values() if len(r) == 2]
assert len(pairs) == 1000, len(pairs)
a0, a1 = (np.array([p[v][0] for p in pairs]) for v in ("V0", "V1"))
b = int(((a0 < 0.7) & (a1 >= 0.7)).sum())  # V0-only dips
c = int(((a1 < 0.7) & (a0 >= 0.7)).sum())  # V1-only dips
n = b + c
p = min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0  # exact two-sided McNemar
for v, a in (("V0 filter", a0), ("V1 momentum 0.5", a1)):
    hon = np.mean([q[v.split()[0]][1] for q in pairs])
    skip = sum(q[v.split()[0]][2] for q in pairs)
    print(f"{v:16} dips {int((a < 0.7).sum()):3}  worst {a.min():.3f}  5th pct {np.percentile(a, 5):.3f}  "
          f"median {np.median(a):.3f}  mean {a.mean():.3f}  honest rejected {hon:.1%}  rounds skipped {skip}")
print(f"discordant dips: V0 only {b}, V1 only {c}, exact McNemar p = {p:.4f}")
fewer = c < b and p < 0.05
median_ok = np.median(a1) >= np.median(a0) - 0.01
print(f"1. fewer V1 dips, p < 0.05: {'PASS' if fewer else 'FAIL'}")
print(f"2. V1 median >= V0 median - 0.01: {'PASS' if median_ok else 'FAIL'}")
print("stage 1b:", "PASS -> stage 2" if fewer and median_ok else "FAIL -> damping dropped")
