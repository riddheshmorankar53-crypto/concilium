"""Candidate second score vs gap E attacks on the CNN (plaintext). python two_score.py <seed> [defenses...]
Per-class references r0, r1 (updates on the root set's class-c samples). Unit vectors e0, e1.
  f = e0 + e1   feature direction (orthogonal to the class axis)
  a = e1 - e0   class axis
Defenses:
  filter   current: keep u.ref > 0, clip to median
  feat     keep u.f > 0, clip to median (= gap E's 'unit per-class sum', tried: collapsed 2/8 seeds)
  balance  feat, then scale down the kept updates that push along a against the root's direction
           (sign of ref.a) until their total class push is at most the aligned ones'
All rules use only u.ref, u.r0, u.r1, u.u (+ the committee's own vectors), so they fit the protocol:
4 revealed scalars per participant instead of 2."""
import sys

import numpy as np

sys.path.insert(0, ".")
import gap_e
from fl_core import N_CLIENTS, N_MAL, keep

np.seterr(over="ignore", invalid="ignore")


def unit(v):
    return v / np.linalg.norm(v)


def aggregate(defense, deltas, ref, r0, r1):
    if defense == "filter":
        return gap_e.aggregate("filter", deltas, ref)
    e0, e1 = unit(r0), unit(r1)
    f, a = e0 + e1, e1 - e0
    kept = keep([d @ f for d in deltas], [d @ d for d in deltas])
    if defense == "balance" and kept:
        want = np.sign(ref @ a)
        c = {i: s * (deltas[i] @ a) * want for i, s in kept.items()}  # > 0: pushes the way the root says
        pro, con = sum(v for v in c.values() if v > 0), -sum(v for v in c.values() if v < 0)
        beta = 1.0 if con <= pro else np.floor(pro / con * 1024) / 1024
        kept = {i: s * (beta if c[i] < 0 else 1.0) for i, s in kept.items()}
        kept = {i: s for i, s in kept.items() if s > 0}
    if len(kept) < 2:
        return np.zeros_like(ref), list(kept)
    return np.mean([s * deltas[i] for i, s in kept.items()], axis=0), list(kept)


def run(t, defense, kind, param, roots):
    w, mal_kept, honest_rej, skipped = t["w0"].copy(), 0, 0, 0
    first = 0 if kind == "none" else N_MAL
    for _ in range(t["rounds"]):
        ref = t["update"](w, *t["root"])
        r0, r1 = (t["update"](w, *r) for r in roots) if defense != "filter" else (None, None)
        deltas = [t["update"](w, X, y) for X, y in t["clients"]]
        if kind != "none":
            deltas[:N_MAL] = gap_e.craft(t, kind, param, w)
        agg, kept = aggregate(defense, deltas, ref, r0, r1)
        mal_kept += sum(i < first for i in kept)
        honest_rej += sum(i not in kept for i in range(first, N_CLIENTS))
        skipped += len(kept) < 2
        w = w + agg
    R = t["rounds"]
    return (*t["evaluate"](w), mal_kept / (R * N_MAL), honest_rej / (R * (N_CLIENTS - first)), skipped)


if __name__ == "__main__":
    seed = int(sys.argv[1])
    defenses = sys.argv[2:] or ["filter", "feat", "balance"]
    t = gap_e.cnn_task(seed)
    Xr, yr = t["root"]
    roots = [(Xr[yr == c], yr[yr == c]) for c in (0, 1)]
    attacks = [("none", None), ("sign-flip x10", None), ("ALIE", None)] + \
        [("inside, hide", p) for p in ((45, 1.0, "hide"), (70, 2.9, "hide"), (85, 2.9, "hide"))] + \
        [("inside, flip", (70, 2.9, "flip"))]
    for kind, param in attacks:
        label = kind if param is None else f"{kind} {param[0]}deg {param[1]}x"
        for d in defenses:
            acc, rec, mal, hon, skip = run(t, d, kind, param, roots)
            print(f"seed {seed} | {label:24} | {d:8} | {acc:.3f} / {rec:.3f} | mal kept {mal:.0%} honest rej {hon:.0%} skipped {skip}", flush=True)
