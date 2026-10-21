"""The filter on standard benchmarks, in plaintext: MNIST and Fashion-MNIST, a 10-class CNN (14,410 parameters,
inside what the encrypted protocol handles), 10 clients of which 3 attack, IID or non-IID (Dirichlet 0.5 per class).

Plaintext is enough for accuracy: the encrypted protocol matches the plaintext filter round by round (deploy.py's
replay check). Defenses: none (FedAvg), the filter (fl_core.keep: angle to the committee's secret reference, clip
to the median length), exact FLTrust, and the usual plaintext baselines, which need to see every update: Krum
(Blanchard et al. 2017, f = 3), coordinate-wise trimmed mean (drop 3 per side) and coordinate-wise median
(Yin et al. 2018). Attacks: gap_e's sign-flip x10 and ALIE, label flipping (attackers train on 9 - y), and
gap_g's optimizing attackers (agr-tailored std, min-max, min-sum, feedback, white-box), pushing toward class 0
where they need a harmful direction. Reported: test accuracy and the worst class's recall (a hiding attack shows
there first). Every run reseeds torch, so the defenses in a cell see the same training randomness.
CIFAR-10 (`cifar`): a 3-channel CNN of 40,682 parameters, 2,500 images per client; GPU recommended.
    python eval_mnist.py <mnist|fashion|cifar> <iid|noniid> <seed> [attack index...]   one JSON line per (attack, defense)
    python eval_mnist.py table <files...>                      mean over seeds, as a markdown table
    python eval_mnist.py data <mnist|fashion|cifar>            only download the data (for tasks/mnist.py)
Data comes from OpenML, CIFAR-10 from Hugging Face (cached as <name>.npz here; train first, the last 10,000 test). ~1.5 h per command on a 4-core CPU.
"""
import json
import os
import sys

import numpy as np

N_CLIENTS, N_MAL, ROUNDS, PER_CLIENT, ALPHA = 10, 3, 30, 600, 0.5
ROOT_SIZE = int(os.environ.get("ROOT", 100))  # the committee's root set; ROOT=500 eval_mnist.py ...: a bigger one
DEFENSES = ("FedAvg", "filter", "FLTrust", "Krum", "trimmed mean", "median")
ATTACKS = [("none", None), ("sign-flip x10", None), ("ALIE", None), ("label flip", None),
           ("agr-tailored", "std"), ("min-max", None), ("min-sum", None), ("feedback", "hide"), ("white-box", "hide")]
OPENML = {"mnist": "mnist_784", "fashion": "Fashion-MNIST"}


def data(name):
    """-> X, y from OpenML, cached as <name>.npz in this folder (so containers can read it offline)."""
    try:
        d = np.load(f"{name}.npz")
        return d["X"], d["y"]
    except FileNotFoundError:
        if name == "cifar":  # Hugging Face's copy (uoft-cs/cifar10): the original server is slow
            import io

            import pandas as pd
            from PIL import Image
            hf = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{}-00000-of-00001.parquet"
            df = pd.concat([pd.read_parquet(hf.format(part)) for part in ("train", "test")])
            X = np.stack([np.asarray(Image.open(io.BytesIO(im["bytes"]))).transpose(2, 0, 1).ravel() for im in df["img"]])
            y = df["label"].to_numpy().astype(np.uint8)
            np.savez_compressed("cifar.npz", X=X, y=y)
            return X, y
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml(OPENML[name], version=1, return_X_y=True, as_frame=False, parser="liac-arff")
        X, y = X.astype(np.uint8), y.astype(np.uint8)
        np.savez_compressed(f"{name}.npz", X=X, y=y)
        return X, y


def task(name, non_iid, seed):
    import torch
    from torch import nn

    import gap_b_cnn as c
    rng = np.random.default_rng(seed)
    X, y = data(name)
    cifar = name == "cifar"
    X = torch.tensor(X.reshape(-1, 3, 32, 32) / 255.0 - 0.5 if cifar else X.reshape(-1, 1, 28, 28) / 255.0,
                     dtype=torch.float32, device=c.DEV)  # GPU if there is one
    y = torch.tensor(y.astype(int), dtype=torch.long, device=c.DEV)
    train, test = np.arange(len(y) - 10000), np.arange(len(y) - 10000, len(y))  # the original split order
    pool = rng.permutation(train)[:N_CLIENTS * (2500 if cifar else PER_CLIENT) + ROOT_SIZE]
    root, rest = pool[:ROOT_SIZE], pool[ROOT_SIZE:]
    if non_iid:  # each class split across clients with Dirichlet(ALPHA) proportions
        shards = [[] for _ in range(N_CLIENTS)]
        for k in range(10):
            idx = rest[y[rest].cpu().numpy() == k]
            cuts = (np.cumsum(rng.dirichlet([ALPHA] * N_CLIENTS)) * len(idx)).astype(int)[:-1]
            for s, part in zip(shards, np.split(idx, cuts)):
                s += part.tolist()
    else:
        shards = np.array_split(rest, N_CLIENTS)
    torch.manual_seed(seed)
    net = (nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3), nn.ReLU(),
                         nn.MaxPool2d(2), nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
                         nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 10)) if cifar else
           nn.Sequential(nn.Conv2d(1, 8, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 3), nn.ReLU(),
                         nn.MaxPool2d(2), nn.Flatten(), nn.Linear(400, 32), nn.ReLU(), nn.Linear(32, 10))).to(c.DEV)
    w0 = c.parameters_to_vector(net.parameters()).detach().double().cpu().numpy()
    hide = np.zeros_like(w0)  # output bias toward class 0, away from the rest
    hide[-10:] = -1 / 9
    hide[-10] = 1.0

    def update(w, X, y):
        return c.local_update(net, w, X, y)

    def evaluate(w):
        if not np.isfinite(w).all():
            return 0.0, 0.0
        c.vector_to_parameters(torch.tensor(w, dtype=torch.float32, device=c.DEV), net.parameters())
        with torch.no_grad():
            pred = net(X[test]).argmax(1)
        recall = [(pred[y[test] == k] == k).float().mean().item() for k in range(10)]
        return (pred == y[test]).float().mean().item(), min(recall)

    return dict(name=f"{name} {'non-IID' if non_iid else 'IID'} seed {seed}", clients=[(X[s], y[s]) for s in shards],
                root=(X[root], y[root]), w0=w0, update=update, evaluate=evaluate, hide=hide, rounds=ROUNDS)


def aggregate(defense, deltas, ref):
    from gap_e import aggregate as gap_e_aggregate
    if defense in ("filter", "FLTrust"):
        return gap_e_aggregate(defense, deltas, ref)
    D = np.array(deltas)
    if defense == "FedAvg":
        return D.mean(axis=0), list(range(len(D)))
    if defense == "Krum":  # the update closest to its n - f - 2 nearest neighbours
        d = ((D[:, None] - D[None]) ** 2).sum(axis=2)
        score = np.sort(d, axis=1)[:, 1:len(D) - N_MAL - 1].sum(axis=1)
        return D[np.argmin(score)], [int(np.argmin(score))]
    if defense == "trimmed mean":
        return np.sort(D, axis=0)[N_MAL:len(D) - N_MAL].mean(axis=0), None
    if defense == "median":
        return np.median(D, axis=0), None
    raise ValueError(defense)


def craft(t, kind, param, w, state):
    import gap_e
    import gap_g
    if kind in ("sign-flip x10", "ALIE"):
        return gap_e.craft(t, kind, param, w)
    if kind == "label flip":
        return [t["update"](w, X, 9 - y) for X, y in t["clients"][:N_MAL]]
    return gap_g.craft(t, kind, param, w, state)


def run(t, defense, kind, param, seed):
    """-> accuracy, worst-class recall, share of poisoned updates kept, share of honest rejected (None where the
    defense keeps no per-update decision)"""
    import torch
    torch.manual_seed(seed)
    w, mal_kept, honest_rej, decided, state = t["w0"].copy(), 0, 0, 0, {}
    attacking = kind != "none"
    for _ in range(t["rounds"]):
        ref = t["update"](w, *t["root"])
        deltas = [t["update"](w, X, y) for X, y in t["clients"]]
        if attacking:
            state["ref"], state["honest_norms"] = ref, [np.linalg.norm(d) for d in deltas]  # white-box only
            deltas[:N_MAL] = craft(t, kind, param, w, state)
        agg, kept = aggregate(defense, deltas, ref)
        if kept is not None:
            decided += 1
            mal_kept += sum(i < N_MAL for i in kept) if attacking else 0
            honest_rej += sum(i not in kept for i in range(N_MAL if attacking else 0, N_CLIENTS))
        w = w + agg
    honest = N_CLIENTS - (N_MAL if attacking else 0)
    acc, worst = t["evaluate"](w)
    return acc, worst, (mal_kept / (decided * N_MAL) if decided and attacking else None), \
        (honest_rej / (decided * honest) if decided else None)


def table(paths):
    rows = [json.loads(line) for p in paths for line in open(p)]
    for setting in sorted({r["setting"] for r in rows}):
        rs = [r for r in rows if r["setting"] == setting]
        seeds = sorted({r["seed"] for r in rs})
        print(f"\n### {setting} (accuracy / worst-class recall, mean of seeds {seeds})\n")
        print("| attack | " + " | ".join(DEFENSES) + " |\n|---|" + "---|" * len(DEFENSES))
        for kind, param in ATTACKS:
            label = kind if param is None else f"{kind} {param}"
            cells = []
            for d in DEFENSES:
                c = [r for r in rs if r["attack"] == label and r["defense"] == d]
                cells.append(f"{np.mean([r['acc'] for r in c]):.3f} / {np.mean([r['worst'] for r in c]):.3f}"
                             + (f" (min {min(r['acc'] for r in c):.2f})" if len(c) > 1 else "") if c else "-")
            print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    np.seterr(over="ignore", invalid="ignore")
    if sys.argv[1] == "table":
        table(sys.argv[2:])
        sys.exit()
    if sys.argv[1] == "data":
        sys.exit(print(data(sys.argv[2])[0].shape))
    name, dist, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))
    t = task(name, dist == "noniid", seed)
    for kind, param in [ATTACKS[int(i)] for i in sys.argv[4:]] or ATTACKS:
        for d in DEFENSES:
            acc, worst, mal, hon = run(t, d, kind, param, seed)
            label = kind if param is None else f"{kind} {param}"
            print(json.dumps(dict(setting=f"{name} {dist}" + (f" root {ROOT_SIZE}" if ROOT_SIZE != 100 else ""), seed=seed, attack=label, defense=d, acc=acc, worst=worst,
                                  poisoned_kept=mal, honest_rejected=hon)), flush=True)
