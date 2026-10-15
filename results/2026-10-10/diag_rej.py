"""Who does the angle check reject on the non-IID CNN, and would per-class references help?
Per round (current filter, no attack): each client's pneumonia share, cos(u, ref), cos(u, r0), cos(u, r1),
where r_c = update on the root set's class-c samples. Also: how often -u passes each rule (attacker room)."""
import sys

import numpy as np

sys.path.insert(0, ".")
import gap_e
from fl_core import keep

np.seterr(over="ignore", invalid="ignore")
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
t = gap_e.cnn_task(seed)
Xr, yr = t["root"]
roots = [(Xr[yr == c], yr[yr == c]) for c in (0, 1)]
share = [float(y.float().mean()) for _, y in t["clients"]]
print("seed", seed, "pneumonia share", np.round(share, 2), "root", [len(r[1]) for r in roots])
cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
w = t["w0"].copy()
rows = []
for rnd in range(t["rounds"]):
    ref = t["update"](w, *t["root"])
    rc = [t["update"](w, *r) for r in roots]
    deltas = [t["update"](w, X, y) for X, y in t["clients"]]
    kept = keep([d @ ref for d in deltas], [d @ d for d in deltas])
    for i, d in enumerate(deltas):
        rows.append((rnd, i, share[i], cos(d, ref), cos(d, rc[0]), cos(d, rc[1]), i in kept))
    w = w + np.mean([s * deltas[i] for i, s in kept.items()], axis=0) if len(kept) >= 2 else w
    print(rnd, "rejected", sorted(set(range(10)) - set(kept)), "cos(r0,r1) %.2f cos(ref,r0) %.2f cos(ref,r1) %.2f"
          % (cos(*rc), cos(ref, rc[0]), cos(ref, rc[1])), flush=True)
print("final", t["evaluate"](w))
R = np.array(rows, dtype=float)
sh, cr, c0, c1, k = R[:, 2], R[:, 3], R[:, 4], R[:, 5], R[:, 6].astype(bool)
print(f"rejected {(~k).mean():.0%}; by client share:")
for i in range(10):
    m = R[:, 1] == i
    print(f"  client {i} share {share[i]:.2f} rejected {(~k[m]).mean():.0%}  mean cos ref {cr[m].mean():+.2f} r0 {c0[m].mean():+.2f} r1 {c1[m].mean():+.2f}")
rules = {"ref>0": cr > 0, "r0>0 or r1>0": (c0 > 0) | (c1 > 0), "r0+r1 (sum of cos)>0": c0 + c1 > 0,
         "ref>0 or (r0>0 and r1>0)": (cr > 0) | ((c0 > 0) & (c1 > 0))}
neg = {"ref>0": -cr > 0, "r0>0 or r1>0": (-c0 > 0) | (-c1 > 0), "r0+r1 (sum of cos)>0": -c0 - c1 > 0,
       "ref>0 or (r0>0 and r1>0)": (-cr > 0) | ((-c0 > 0) & (-c1 > 0))}
for name in rules:
    print(f"rule {name:28} honest pass {rules[name].mean():.0%}, sign-flip of an honest update passes {neg[name].mean():.0%}")
