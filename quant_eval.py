"""Plaintext check: what does ternary quantization of the updates cost in accuracy?

TernGrad-style: each participant sends s * t with s = max|u| and t in {-1, 0, +1}^n, rounding
stochastically so the expectation is u. The committee's ref update stays full precision. Attackers
quantize too (the protocol would have to enforce the ternary format), after crafting their update.
    python quant_eval.py        # LR IID / non-IID, CNN non-IID; 5 seeds each
"""
import numpy as np

from fl_core import N_CLIENTS, N_MAL
from gap_e import aggregate, cnn_task, craft, lr_task

SEEDS = range(5)


def ternary(u, rng):
    s = np.abs(u).max()
    return u if s == 0 else s * np.sign(u) * (rng.random(u.shape) < np.abs(u) / s)


def run(t, quant, attack, rng):
    w = t["w0"].copy()
    for _ in range(t["rounds"]):
        ref = t["update"](w, *t["root"])
        deltas = [t["update"](w, X, y) for X, y in t["clients"]]
        if attack:
            deltas[:N_MAL] = craft(t, "sign-flip x10", None, w)
        if quant:
            deltas = [ternary(d, rng) for d in deltas]
        w = w + aggregate("none" if not attack else "filter", deltas, ref)[0]
    return t["evaluate"](w)[:2]


def table(make, name):
    print(f"\n=== {name}, {N_CLIENTS} clients, {len(SEEDS)} seeds ===")
    print(f"{'setting':34} {'acc mean':>8} {'min':>6} {'recall':>7}")
    for attack in (False, True):
        for quant in (False, True):
            rs = np.array([run(make(s), quant, attack, np.random.default_rng(s)) for s in SEEDS])
            label = f"{'ternary' if quant else 'float  '}, {'sign-flip x10 + filter' if attack else 'no attack'}"
            print(f"{label:34} {rs[:, 0].mean():8.3f} {rs[:, 0].min():6.3f} {rs[:, 1].mean():7.3f}", flush=True)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    u = rng.normal(size=100_000)
    q = np.mean([ternary(u, rng) for _ in range(200)], axis=0)
    assert set(np.unique(ternary(u, rng) / np.abs(u).max())) <= {-1.0, 0.0, 1.0}
    assert np.abs(q - u).mean() < 0.1 * np.abs(u).max(), "ternary() must be unbiased"
    table(lambda s: lr_task(False), "breast cancer LR, IID")
    table(lambda s: lr_task(True), "breast cancer LR, non-IID")
    table(cnn_task, "chest X-ray CNN, non-IID")
