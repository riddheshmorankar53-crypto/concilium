"""DP noise on each round's sum (decision D follow-up, 2026-10-11). Range proofs bound every update by B (= 1), and
the filter only scales updates down, so one participant moves the sum by at most B: Gaussian noise of std z*B per
coordinate on the sum is the Gaussian mechanism (per round: eps = sqrt(2 ln(1.25/delta)) / z, for z >= ~0.5).
Measures, CNN, filter as in fl_core.keep, honest updates clipped to B like deploy:
  - training: accuracy with no attack, sign-flip x10, inside-flip (full roster);
  - isolation: roster = 3 colluders (honest-looking updates) + 3 honest; in each decrypted round with exactly one
    honest update kept, the colluders' reconstruction k*published - their own terms (clip factors given to them)
    vs that update: cosine similarity (1 = recovered, ~0 = hidden).
    ~/ppfl-gpu/Scripts/python results/2026-10-11/dp_measure.py <seed> <draws> [z ...]
Output: seed|draw|z|setting|acc|recall|isolated rounds|median cosine|max cosine"""
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
import gap_e  # noqa: E402
from fl_core import N_CLIENTS, N_MAL, keep  # noqa: E402

np.seterr(over="ignore", invalid="ignore")
B, ZS = 1.0, (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)
SETTINGS = [("none", None, "full"), ("sign-flip x10", None, "full"), ("inside, flip", (70, 2.9, "flip"), "full"),
            ("isolation", None, "min")]


def clip(u):
    n = np.linalg.norm(u)
    return u if n <= B else u * (B / n)


def run(t, kind, param, roster_mode, z, rng):
    w, cosines = t["w0"].copy(), []
    for _ in range(t["rounds"]):
        ref = t["update"](w, *t["root"])
        deltas = [clip(t["update"](w, X, y)) for X, y in t["clients"]]  # honest participants clip to B
        if kind not in ("none", "isolation"):
            deltas[:N_MAL] = gap_e.craft(t, kind, param, w)  # attackers would fail the range proof if over B: clip
            deltas[:N_MAL] = [clip(d) for d in deltas[:N_MAL]]
        roster = list(range(N_CLIENTS)) if roster_mode == "full" else \
            list(range(N_MAL)) + sorted(rng.choice(range(N_MAL, N_CLIENTS), 3, replace=False).tolist())
        ds = [deltas[c] for c in roster]
        kept = {roster[i]: s for i, s in keep([d @ ref for d in ds], [d @ d for d in ds]).items()}
        if len(kept) < 2:
            continue
        total = sum(s * deltas[c] for c, s in kept.items()) + rng.normal(0, z * B, len(w))
        if kind == "isolation" and sum(c >= N_MAL for c in kept) == 1:
            (h, s), = [(c, s) for c, s in kept.items() if c >= N_MAL]
            guess = total - sum(s * deltas[c] for c, s in kept.items() if c < N_MAL)  # = s*u_h + noise
            cosines.append(float(guess @ deltas[h] / (np.linalg.norm(guess) * np.linalg.norm(deltas[h]) + 1e-300)))
        w = w + total / len(kept)
    return (*t["evaluate"](w), cosines)


if __name__ == "__main__":
    seed, draws = int(sys.argv[1]), int(sys.argv[2])
    t = gap_e.cnn_task(seed)
    for rep in range(draws):
        for z in [float(x) for x in sys.argv[3:]] or ZS:
            for kind, param, roster_mode in SETTINGS:
                torch.manual_seed(1000 * rep + 7)
                rng = np.random.default_rng(1000 * rep + seed)
                acc, rec, cos = run(t, kind, param, roster_mode, z, rng)
                med, mx = (f"{np.median(cos):.3f}", f"{max(cos):.3f}") if cos else ("-", "-")
                print(f"{seed}|{rep}|{z}|{kind}|{acc:.3f}|{rec:.3f}|{len(cos)}|{med}|{mx}", flush=True)
