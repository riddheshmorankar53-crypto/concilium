"""Would a secret-sharing design (PINE-style validity proofs) keep the filter's protection? Plaintext, CNN.

PINE (Rothblum et al., USENIX Security 2024) lets the committee check |u| <= B for a PUBLIC bound B on a
secret-shared update, revealing only valid/invalid; it catches the norm wraparound of gap F. It can't do
fl_core.keep's median (data-dependent) or its angle check (u.ref > 0, ref secret to the committee). So:
  rules   (gap E attacks, attackers estimate ref from their own data; clipping everyone to B simulates
           honest participants clipping themselves and attackers sending at most B):
    med+ang   fl_core.keep: clip to the median length, keep if u.ref > 0 (today's filter)
    med       the same without the angle check
    ref1+ang  clip to B = |ref| (public per round), keep if u.ref > 0     <- what PINE + a comparison can do
    ref1      clip to B = |ref|, no angle check                           <- what PINE alone can do
    ref3      clip to B = 3|ref|, no angle check
  known   ref1+ang when ref is PUBLISHED (so participants could prove u.ref >= 0 in the same proof):
          attackers then know ref exactly and push at angle theta from it, at length B
    python eval_ss.py            # everything, 5 seeds, in parallel (~40 min on 16 cores)
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

import fl_core
import gap_e
from gap_e import unit

RULES = ("med+ang", "med", "ref1+ang", "ref1", "ref3")
ATTACKS = [("none", None), ("sign-flip x10", None), ("ALIE", None), ("inside, hide", (45, 1.0, "hide")),
           ("inside, hide", (70, 2.9, "hide")), ("inside, hide", (85, 2.9, "hide")), ("inside, flip", (70, 2.9, "flip"))]
KNOWN = [(th, push) for push in ("hide", "flip") for th in (45, 70, 85, 89.9)]
SEEDS = range(5)


def rule_run(job):
    """One gap E run with a PINE-compatible rule -> accuracy."""
    rule, seed, a = job
    kind, param = ATTACKS[a]

    def aggregate(defense, deltas, ref):
        dots, norms = [d @ ref for d in deltas], [d @ d for d in deltas]
        if rule.startswith("med"):
            kept = fl_core.keep(dots if rule == "med+ang" else [1.0] * len(deltas), norms)
        else:
            b2 = int(rule[3]) ** 2 * (ref @ ref)
            kept = {i: min(1.0, np.sqrt(b2 / n)) for i, (d, n) in enumerate(zip(dots, norms))
                    if n > 0 and (d > 0 or not rule.endswith("+ang"))}
        if len(kept) < 2:
            return np.zeros_like(ref), list(kept)
        return np.mean([s * deltas[i] for i, s in kept.items()], axis=0), list(kept)

    gap_e.aggregate = aggregate
    return gap_e.run(gap_e.cnn_task(seed), "filter", kind, param)[0]


def known_run(job):
    """ref1+ang against attackers who know ref: harmful push h at angle theta from ref, length B -> (acc, recall)."""
    seed, a = job
    theta, push = KNOWN[a]
    t, n = gap_e.cnn_task(seed), fl_core.N_MAL
    w = t["w0"].copy()
    for _ in range(t["rounds"]):
        ref = t["update"](w, *t["root"])
        b = np.linalg.norm(ref)
        deltas = [t["update"](w, X, y) for X, y in t["clients"]]
        own = t["clients"][:n]
        h = np.mean([t["update"](w, X, 1 - y) for X, y in own], axis=0) if push == "flip" else t["hide"].copy()
        h = h - (h @ ref) / (ref @ ref) * ref  # exactly the part the angle check can't see
        deltas[:n] = [b * (np.cos(np.radians(theta)) * unit(ref) + np.sin(np.radians(theta)) * unit(h))] * n
        kept = {i: min(1.0, b / np.linalg.norm(d)) for i, d in enumerate(deltas) if d @ ref > 0}
        if len(kept) >= 2:
            w = w + np.mean([s * deltas[i] for i, s in kept.items()], axis=0)
    return t["evaluate"](w)


def init():
    import torch
    torch.set_num_threads(1)
    np.seterr(all="ignore")


if __name__ == "__main__":
    with Pool(os.cpu_count(), initializer=init) as pool:
        jobs = [(r, s, a) for r in RULES for s in SEEDS for a in range(len(ATTACKS))]
        acc = dict(zip(jobs, pool.map(rule_run, jobs)))
        kjobs = [(s, a) for s in SEEDS for a in range(len(KNOWN))]
        known = dict(zip(kjobs, pool.map(known_run, kjobs)))
    print("Gap E attacks (attackers estimate ref). Worst accuracy over attacks, per seed:")
    print(f"{'rule':10} {'seed ' + ' '.join(map(str, SEEDS)):>36} {'mean':>6} {'attacked runs < 0.7':>20} {'no attack':>12}")
    summary = {}
    for r in RULES:
        worst = [min(acc[r, s, a] for a in range(1, len(ATTACKS))) for s in SEEDS]
        bad = sum(acc[r, s, a] < 0.7 for s in SEEDS for a in range(1, len(ATTACKS)))
        clean = [acc[r, s, 0] for s in SEEDS]
        summary[r] = np.mean(worst)
        print(f"{r:10} {' '.join(f'{x:.3f}' for x in worst):>36} {np.mean(worst):6.3f} {bad:20} "
              f"{min(clean):.3f}-{max(clean):.3f}")
    print(f"\nref1+ang when ref is published (attackers know it exactly), accuracy / pneumonia recall:")
    for a, (th, push) in enumerate(KNOWN):
        print(f"  {push} {th:>4} deg: " + "  ".join(f"{known[s, a][0]:.3f}/{known[s, a][1]:.3f}" for s in SEEDS))
    known_bad = sum(known[s, a][0] < 0.7 for s in SEEDS for a in range(len(KNOWN)))
    print(f"  runs < 0.7: {known_bad} of {len(kjobs)}")
    # the conclusions this evaluation rests on (5 seeds; single cells vary ~0.1 between runs)
    assert summary["ref1"] < summary["ref1+ang"] - 0.1, "without the angle check, bounded sign-flip should still hurt"
    assert abs(summary["ref1+ang"] - summary["med+ang"]) < 0.1, "a public bound |ref| should be about as good as the median"
    assert known_bad >= 4, "publishing ref should clearly help the attackers"
