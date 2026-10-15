"""Gap G: optimizing attackers. gap_e.py's attacks are fixed recipes; these search for the most harmful update
the filter still keeps, and adapt each round.

The colluding attackers (clients 0..N_MAL-1) know the filter's rule but not the committee's secret reference,
and see their own data plus the published model each round:
  agr-tailored  Shejwalkar & Houmansadr (NDSS 2021): mean of their own honest updates + gamma x a perturbation
                (inverse unit vector, inverse sign, inverse std), gamma as large as their simulation of the
                filter keeps, by binary search (their own mean stands in for the reference)
  min-max/sum   same paper, needs no knowledge of the filter: gamma as large as keeps the update no farther
                from their honest updates than those are from each other (max distance / sum of squares)
  feedback      tailored to our filter's one boundary, the angle check: tilt the update from their own mean toward
                a harmful direction; from how the published model moved, infer whether they were kept, and tilt
                further (5 deg) if so, back off (10 deg) if not. Sent 10x long: the clip shortens it anyway, and
                3 long updates lift the median the clip uses (to between the 5th and 6th of 7 honest lengths).
                Its inference is crude: honest updates also move the model along the push (flip: against it)
  white-box     a bound, not an attack on our design: knows the secret reference and sits at 89.5 deg from it
                (the filter's own boundary), 10x long. The most harm along these pushes the filter lets through
Pushes as in gap_e: flip = their label-flipped update, hide = output bias toward the benign class.
    python gap_g.py            # breast cancer LR, IID and non-IID (seconds)
    python gap_g.py cnn [seed] # chest X-ray CNN (~20 min); RESEARCH.md "Gap G" has seeds 0-4 (OMP_NUM_THREADS=2 each)
"""
import sys

import numpy as np

from fl_core import N_CLIENTS, N_MAL
from gap_e import aggregate, cnn_task, lr_task, unit

DEFENSES = ("none", "filter", "FLTrust")


def push_dir(t, w, own, g, push):
    """The harmful direction, minus its part along g (the part the angle check can't see)."""
    h = np.mean([t["update"](w, X, 1 - y) for X, y in own], axis=0) if push == "flip" else t["hide"].copy()
    return unit(h - (h @ g) / (g @ g) * g)


def largest(ok, hi, steps=30):
    """Largest gamma in [0, hi] with ok(gamma), by binary search (ok holds at 0)."""
    lo = 0.0
    for _ in range(steps):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def craft(t, kind, param, w, state):
    own = t["clients"][:N_MAL]
    honest = np.array([t["update"](w, X, y) for X, y in own])
    mu, med = honest.mean(axis=0), float(np.median(np.linalg.norm(honest, axis=1)))
    if kind == "agr-tailored":
        p = {"unit": -unit(mu), "sign": -np.sign(mu), "std": -honest.std(axis=0)}[param]
        gamma = largest(lambda g: (mu + g * p) @ mu > 0, 100 * np.linalg.norm(mu) / max(np.linalg.norm(p), 1e-12))
        return [mu + gamma * p] * N_MAL
    if kind in ("min-max", "min-sum"):
        p = -honest.std(axis=0)
        d = np.linalg.norm(honest[:, None] - honest[None], axis=2)
        if kind == "min-max":
            ok = lambda g: np.linalg.norm(honest - (mu + g * p), axis=1).max() <= d.max()
        else:
            ok = lambda g: (np.linalg.norm(honest - (mu + g * p), axis=1) ** 2).sum() <= (d ** 2).sum(axis=1).max()
        return [mu + largest(ok, 100 * np.linalg.norm(mu) / max(np.linalg.norm(p), 1e-12)) * p] * N_MAL
    if kind == "feedback":
        h = push_dir(t, w, own, mu, param)
        if "prev" in state:  # were we kept last round? our push shows in the model's move along h
            moved = (w - state["prev"]) @ state["h"]
            expected = N_MAL * state["length"] * np.sin(state["theta"]) / N_CLIENTS
            state["theta"] += np.radians(5) if moved > expected / 2 else -np.radians(10)
            state["theta"] = float(np.clip(state["theta"], np.radians(5), np.radians(89.5)))
        state.setdefault("theta", np.radians(45))
        state["prev"], state["h"], state["length"] = w.copy(), h, med  # what shows if kept: clipped to ~median
        # 10x long: clipped anyway, and 3 long updates lift the median the clip uses (still between two honest lengths)
        return [10 * med * (np.cos(state["theta"]) * unit(mu) + np.sin(state["theta"]) * h)] * N_MAL
    if kind == "white-box":
        ref, others = state["ref"], state["honest_norms"]
        h = push_dir(t, w, own, ref, param)
        theta = np.radians(89.5)
        return [10 * max(others) * (np.cos(theta) * unit(ref) + np.sin(theta) * h)] * N_MAL
    raise ValueError(kind)


def run(t, defense, kind, param):
    """-> accuracy, recall, share of poisoned updates kept, share of honest rejected, rounds skipped, final angle"""
    w, mal_kept, honest_rej, skipped, state = t["w0"].copy(), 0, 0, 0, {}
    attacking = kind != "none"
    for _ in range(t["rounds"]):
        ref = t["update"](w, *t["root"])
        deltas = [t["update"](w, X, y) for X, y in t["clients"]]
        if attacking:
            state["ref"], state["honest_norms"] = ref, [np.linalg.norm(d) for d in deltas]  # white-box only
            deltas[:N_MAL] = craft(t, kind, param, w, state)
        agg, kept = aggregate(defense, deltas, ref)
        if kept is not None:
            mal_kept += sum(i < N_MAL for i in kept) if attacking else 0
            honest_rej += sum(i not in kept for i in range(N_MAL if attacking else 0, N_CLIENTS))
            skipped += len(kept) < 2
        w = w + agg
    rounds, honest = t["rounds"], N_CLIENTS - (N_MAL if attacking else 0)
    angle = np.degrees(state["theta"]) if "theta" in state else None
    return (*t["evaluate"](w), mal_kept / (rounds * N_MAL), honest_rej / (rounds * honest), skipped, angle)


ATTACKS = [("none", None), ("agr-tailored", "unit"), ("agr-tailored", "sign"), ("agr-tailored", "std"),
           ("min-max", None), ("min-sum", None), ("feedback", "flip"), ("feedback", "hide"),
           ("white-box", "flip"), ("white-box", "hide")]


def table(t):
    print(f"\n=== {t['name']} | accuracy / {t['recall']} (filter: poisoned kept, honest rejected, rounds skipped) ===")
    print(f"{'attack':22}" + "".join(f"{d:>30}" for d in DEFENSES), flush=True)
    out = {}
    for kind, param in ATTACKS:
        label = kind if param is None else f"{kind} {param}"
        cells = []
        for d in DEFENSES:
            acc, rec, mal, hon, skip, angle = run(t, d, kind, param)
            out[label, d] = (acc, rec, angle)
            extra = f" ({mal:.0%}, {hon:.0%}, {skip})" if d == "filter" else ""
            cells.append(f"{acc:.3f} / {rec:.3f}{extra}")
        print(f"{label:22}" + "".join(f"{c:>30}" for c in cells), flush=True)
    angles = {k[0]: v[2] for k, v in out.items() if k[1] == "filter" and v[2] is not None}
    if angles:
        print("feedback attack, final tilt from the attackers' own mean under the filter: "
              + ", ".join(f"{k} {a:.0f} deg" for k, a in angles.items()))
    return out


if __name__ == "__main__":
    np.seterr(over="ignore", invalid="ignore")
    if sys.argv[1:2] == ["cnn"]:
        table(cnn_task(int(sys.argv[2]) if len(sys.argv) > 2 else 0))
    else:
        for non_iid in (False, True):
            out = table(lr_task(non_iid))
            worst = min(acc for (attack, d), (acc, _, _) in out.items() if d == "filter")
            assert worst > 0.9, f"an optimizing attack got through the filter ({worst:.3f})"
            assert out["white-box flip", "none"][0] < 0.7, "the attacks must be real: white-box must break FedAvg"
