"""Gap E: attackers who know the filter.

fl_core's filter keeps an update if it points the same way as the server's reference (u.ref > 0)
and clips it to the median update's length. An attacker who knows that stays just inside:
the colluding attackers' mean honest update g plus as much harmful push h as the angle allows, at the
largest length it expects to pass. They see only their own data (ref is unknown, g stands in for it;
the median of their own honest lengths stands in for the median).
Attacks: sign-flip x10 (fl_core's), ALIE (Baruch et al. 2019: mean - z*std of the attackers' own
honest updates), and "inside" (angle theta from g toward h, length k x median of their own honest lengths):
flip = a label-flip update, hide = a steady push on the output bias that hides the disease.
Defenses: none, the old filter (reject if u.ref <= 0 or |u| > 3|ref|), fl_core's filter, exact FLTrust.
FLTrust needs only u.ref and u.u, which the threshold committee already decrypts, so it costs nothing
extra there and has no polynomial to attack. Like the protocol, the filters skip a round with < 2 kept.
    python gap_e.py            # breast cancer, logistic regression (seconds)
    python gap_e.py cnn [seed] # chest X-ray CNN (~15 min); seed changes the init, the split and the root set
    python gap_e.py reps <seed> <n>  # CNN, no defense vs filter, n training draws per attack (one line per run)
A table cell is one draw of the training randomness, and the torch state carries over between cells; `reps`
reseeds torch before every run, so its runs are independent draws.
"""
import os
import sys

import numpy as np
from scipy.stats import norm

import fl_core
from fl_core import N_CLIENTS, N_MAL, keep
from gap_b import exact_ops, fltrust

DEFENSES = ("none", "old filter", "filter", "FLTrust")
BASE = [("none", None), ("sign-flip x10", None), ("ALIE", None)]
CNN_ATTACKS = BASE + [("inside, hide", p) for p in ((45, 1.0, "hide"), (70, 2.9, "hide"), (85, 2.9, "hide"))] \
    + [("inside, flip", (70, 2.9, "flip"))]


def unit(v):
    return v / np.linalg.norm(v)


def lr_task(non_iid):
    clients, root, (X, y) = fl_core.load(non_iid)
    hide = np.zeros(X.shape[1])
    hide[-1] = 1.0  # bias toward 'benign' (1): malignant cases missed

    def evaluate(w):
        pred = X @ w > 0
        return float((pred == y).mean()), float((pred[y == 0] == 0).mean())

    return dict(name=f"breast cancer LR, {'non-IID' if non_iid else 'IID'}", recall="malignant recall",
                clients=clients, root=root, w0=np.zeros(X.shape[1]), update=fl_core.local_update,
                evaluate=evaluate, hide=hide, rounds=fl_core.ROUNDS)


def cnn_task(seed=0):
    import torch

    import gap_b_cnn as c
    torch.manual_seed(seed)
    clients, root, test = c.load(seed)
    net = c.model()
    w0 = c.parameters_to_vector(net.parameters()).detach().double().cpu().numpy()
    hide = np.zeros_like(w0)
    hide[-2], hide[-1] = 1.0, -1.0  # output bias: +normal, -pneumonia
    return dict(name="chest X-ray CNN, non-IID", recall="pneumonia recall", clients=clients, root=root, w0=w0,
                update=lambda w, X, y: c.local_update(net, w, X, y), evaluate=lambda w: c.evaluate(net, w, *test),
                hide=hide, rounds=c.ROUNDS)


def craft(t, kind, param, w):
    """Malicious clients 0..N_MAL-1 craft updates from their own data only."""
    own = t["clients"][:N_MAL]
    honest = [t["update"](w, X, y) for X, y in own]
    if kind == "sign-flip x10":
        return [-10 * g for g in honest]
    if kind == "ALIE":
        s = N_CLIENTS // 2 + 1 - N_MAL  # benign clients the attackers need on their side
        z = norm.ppf((N_CLIENTS - s) / N_CLIENTS)
        mu, sd = np.mean(honest, axis=0), np.std(honest, axis=0)
        return [mu - z * sd * np.sign(mu)] * N_MAL
    theta, k, push = np.radians(param[0]), param[1], param[2]
    g = np.mean(honest, axis=0)  # colluding attackers' best guess of ref
    h = np.mean([t["update"](w, X, 1 - y) for X, y in own], axis=0) if push == "flip" else t["hide"].copy()
    h = h - (h @ g) / (g @ g) * g  # keep only the part the angle check can't see
    return [k * np.median(np.linalg.norm(honest, axis=1)) * (np.cos(theta) * unit(g) + np.sin(theta) * unit(h))] * N_MAL


def aggregate(defense, deltas, ref):
    if defense == "FLTrust":
        return fltrust(deltas, ref, *exact_ops()), None
    dots, norms = [d @ ref for d in deltas], [d @ d for d in deltas]
    if defense == "none":
        kept = dict.fromkeys(range(len(deltas)), 1.0)
    elif defense == "old filter":  # before the median clip: honest non-IID updates failed it whenever ref was short
        kept = {i: 1.0 for i, (d, n) in enumerate(zip(dots, norms)) if d > 0 and n <= 9 * (ref @ ref)}
    else:
        kept = keep(dots, norms)
    if len(kept) < 2:  # the protocol won't decrypt a "sum" of one update: the round is skipped
        return np.zeros_like(ref), list(kept)
    return np.mean([s * deltas[i] for i, s in kept.items()], axis=0), list(kept)


def run(t, defense, kind, param, momentum=0.0, smooth=1):
    """-> accuracy, recall, share of poisoned updates kept, share of honest updates rejected, rounds skipped.
    Damping variants (overnight 2026-10-10, both off by default): momentum b: v = b v + agg, w += (1 - b) v
    (same step size at steady state); smooth k: the filter scores against the mean of the last k unit root
    updates instead of this round's ref (the committee would keep them as secret state)."""
    w, mal_kept, honest_rej, skipped = t["w0"].copy(), 0, 0, 0
    v, refs = np.zeros_like(w), []
    first_honest = 0 if kind == "none" else N_MAL
    for _ in range(t["rounds"]):
        ref = t["update"](w, *t["root"])
        refs = (refs + [unit(ref)])[-smooth:]
        deltas = [t["update"](w, X, y) for X, y in t["clients"]]
        if kind != "none":
            deltas[:N_MAL] = craft(t, kind, param, w)
        agg, kept = aggregate(defense, deltas, ref if smooth == 1 else np.mean(refs, axis=0))
        if kept is not None:
            mal_kept += sum(i < first_honest for i in kept)
            honest_rej += sum(i not in kept for i in range(first_honest, N_CLIENTS))
            skipped += len(kept) < 2
        if momentum:
            v = momentum * v + agg
            agg = (1 - momentum) * v
        w = w + agg
    rounds = t["rounds"]
    return (*t["evaluate"](w), mal_kept / (rounds * N_MAL), honest_rej / (rounds * (N_CLIENTS - first_honest)), skipped)


def table(t, attacks):
    print(f"\n=== {t['name']} | accuracy / {t['recall']} "
          f"(filters: poisoned kept, honest rejected, rounds skipped) ===")
    print(f"{'attack':24}" + "".join(f"{d:>30}" for d in DEFENSES), flush=True)
    worst = {d: (1.0, 1.0) for d in DEFENSES}
    for kind, param in attacks:
        label = kind if param is None else f"{kind} {param[0]}deg {param[1]}x"
        cells = []
        for d in DEFENSES:
            acc, rec, mal, hon, skip = run(t, d, kind, param)
            if kind != "none":
                worst[d] = (min(worst[d][0], acc), min(worst[d][1], rec))
            cells.append(f"{acc:.3f} / {rec:.3f}" + (f" ({mal:.0%}, {hon:.0%}, {skip})" if "filter" in d else ""))
        print(f"{label:24}" + "".join(f"{c:>30}" for c in cells), flush=True)
    print(f"{'worst case':24}" + "".join(f"{a:.3f} / {r:.3f}".rjust(30) for a, r in worst.values()))
    return worst


if __name__ == "__main__":
    np.seterr(over="ignore", invalid="ignore")
    if sys.argv[1:2] == ["cnn"]:
        table(cnn_task(int(sys.argv[2]) if len(sys.argv) > 2 else 0), CNN_ATTACKS)
    elif sys.argv[1:2] == ["reps"]:
        import torch
        seed, t = int(sys.argv[2]), cnn_task(int(sys.argv[2]))
        for rep in range(int(sys.argv[3])):
            for kind, param in CNN_ATTACKS:
                for d in ("none", "filter"):
                    torch.manual_seed(1000 * rep + 7)
                    acc, rec, mal, hon, skip = run(t, d, kind, param)
                    print(f"seed {seed} draw {rep} | {kind} {param or ''} | {d} | {acc:.3f} / {rec:.3f} | "
                          f"honest rejected {hon:.0%}, poisoned kept {mal:.0%}, skipped {skip}", flush=True)
    elif sys.argv[1:2] == ["damp"]:  # python gap_e.py damp <seed> <first draw> <draws> <attack index>...
        import torch
        seed, t = int(sys.argv[2]), cnn_task(int(sys.argv[2]))
        variants = {"V0": {}, "V1 momentum 0.5": {"momentum": 0.5}, "V2 smooth 3": {"smooth": 3}}
        if os.environ.get("VARIANTS"):  # e.g. VARIANTS=V0,V1 (name prefixes)
            variants = {k: v for k, v in variants.items() if k.split()[0] in os.environ["VARIANTS"].split(",")}
        for rep in range(int(sys.argv[3]), int(sys.argv[3]) + int(sys.argv[4])):
            for kind, param in [CNN_ATTACKS[int(a)] for a in sys.argv[5:]]:
                for name, kw in variants.items():
                    torch.manual_seed(1000 * rep + 7)  # paired: every variant starts from the same draw
                    acc, rec, mal, hon, skip = run(t, "filter", kind, param, **kw)
                    print(f"{seed}|{rep}|{kind} {param or ''}|{name}|{acc:.3f}|{rec:.3f}|{hon:.3f}|{mal:.3f}|{skip}",
                          flush=True)
    else:
        grid = [(f"inside, {push}", (th, k, push)) for push in ("flip", "hide") for th in (45, 70, 85) for k in (1.0, 2.9)]
        for non_iid in (False, True):
            worst = table(lr_task(non_iid), BASE + grid)
            assert worst["filter"][0] > 0.9 and worst["FLTrust"][0] > 0.9, "a filter-aware attack got through"
