"""Kept-update floor (decision D, 2026-10-11): how often would rounds be skipped, and what happens to accuracy,
if a round is decrypted only when at least `floor` updates pass the filter (today: 2)?
Rosters: "full" = all 10 participants; "min" = the 3 attackers + 3 random honest participants (the smallest
roster option D allows, n - f - 1 = 6). Attackers are participants 0..2 (gap_e). CNN, filter as in fl_core.keep.
    ~/ppfl-gpu/Scripts/python results/2026-10-11/floor_measure.py <seed> <draws>   (from the project folder)
Output lines: seed|draw|attack|roster|floor|acc|recall|rounds skipped|mean kept|min honest kept in a decrypted round"""
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
import gap_e  # noqa: E402
from fl_core import N_CLIENTS, N_MAL, keep  # noqa: E402

np.seterr(over="ignore", invalid="ignore")
ATTACKS = [("none", None), ("sign-flip x10", None), ("inside, flip", (70, 2.9, "flip"))]
FLOORS = (2, 4, 5, 6)


def run(t, kind, param, roster_mode, floor, rng):
    w, skipped, kept_n, honest_min = t["w0"].copy(), 0, [], N_CLIENTS
    for _ in range(t["rounds"]):
        ref = t["update"](w, *t["root"])
        deltas = [t["update"](w, X, y) for X, y in t["clients"]]
        if kind != "none":
            deltas[:N_MAL] = gap_e.craft(t, kind, param, w)
        roster = list(range(N_CLIENTS)) if roster_mode == "full" else \
            list(range(N_MAL)) + sorted(rng.choice(range(N_MAL, N_CLIENTS), 3, replace=False).tolist())
        ds = [deltas[c] for c in roster]
        kept = {roster[i]: s for i, s in keep([d @ ref for d in ds], [d @ d for d in ds]).items()}
        kept_n.append(len(kept))
        honest = sum(c >= N_MAL for c in kept) if kind != "none" else len(kept)
        if len(kept) < floor:
            skipped += 1
            continue
        honest_min = min(honest_min, honest)
        w = w + np.mean([s * deltas[c] for c, s in kept.items()], axis=0)
    return (*t["evaluate"](w), skipped, float(np.mean(kept_n)), honest_min)


if __name__ == "__main__":
    seed, draws = int(sys.argv[1]), int(sys.argv[2])
    t = gap_e.cnn_task(seed)
    for rep in range(draws):
        for kind, param in ATTACKS:
            for roster_mode in ("full", "min"):
                for floor in FLOORS:
                    torch.manual_seed(1000 * rep + 7)  # paired across floors and rosters
                    rng = np.random.default_rng(1000 * rep + seed)  # same "min" rosters for every floor
                    acc, rec, skip, mk, hmin = run(t, kind, param, roster_mode, floor, rng)
                    print(f"{seed}|{rep}|{kind}|{roster_mode}|{floor}|{acc:.3f}|{rec:.3f}|{skip}|{mk:.2f}|{hmin}", flush=True)
