"""Gap B on a chest X-ray CNN (PneumoniaMNIST 28x28), non-IID clients (Dirichlet 0.5).

Attack goal: hide pneumonia (push the output bias toward 'normal'). The attacker picks its update
size from the public approximation alone (largest |amplification| just outside the range), so no
simulation of the FL run is needed. CPU is enough (~14k-parameter CNN).
    python -c "from medmnist import PneumoniaMNIST; PneumoniaMNIST(split='train', download=True, root='.')"
    python gap_b_cnn.py
"""
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from gap_b import MAX_T, approx_ops, exact_ops, fltrust, poison

N_CLIENTS, N_MAL, ROUNDS, ROOT_SIZE, ALPHA = 10, 3, 30, 100, 0.5
# the GPU when this torch has CUDA (host experiments); the Docker deploy is CPU-only, where training must be
# deterministic for the replay
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CONFIGS = [(7, 2, 0), (15, 2, 2)]  # plain degree 7, and the strongest baseline from gap_b.py


def model():
    return nn.Sequential(nn.Conv2d(1, 8, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 3), nn.ReLU(),
                         nn.MaxPool2d(2), nn.Flatten(), nn.Linear(400, 32), nn.ReLU(), nn.Linear(32, 2)).to(DEV)


def load(seed=0):
    rng = np.random.default_rng(seed)
    d = np.load("pneumoniamnist.npz")
    t = lambda k: (torch.tensor(d[f"{k}_images"] / 255.0, dtype=torch.float32).unsqueeze(1),
                   torch.tensor(d[f"{k}_labels"].ravel(), dtype=torch.long))
    (Xtr, ytr), (Xva, yva), test = t("train"), t("val"), t("test")
    root = rng.choice(len(Xva), ROOT_SIZE, replace=False)
    # non-IID: each class split across clients with Dirichlet(ALPHA) proportions
    shards = [[] for _ in range(N_CLIENTS)]
    for c in (0, 1):
        idx = rng.permutation(np.where(ytr.numpy() == c)[0])
        cuts = (np.cumsum(rng.dirichlet([ALPHA] * N_CLIENTS)) * len(idx)).astype(int)[:-1]
        for s, part in zip(shards, np.split(idx, cuts)):
            s += part.tolist()
    on = lambda X, y: (X.to(DEV), y.to(DEV))
    return [on(Xtr[s], ytr[s]) for s in shards], on(Xva[root], yva[root]), on(*test)


def local_update(net, w, X, y, epochs=2, lr=0.1, bs=32):
    vector_to_parameters(torch.tensor(w, dtype=torch.float32, device=DEV), net.parameters())
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    for _ in range(epochs):
        for i in torch.randperm(len(y), device=DEV).split(bs):
            opt.zero_grad()
            nn.functional.cross_entropy(net(X[i]), y[i]).backward()
            opt.step()
    return parameters_to_vector(net.parameters()).detach().double().cpu().numpy() - w


def evaluate(net, w, X, y):
    if not np.isfinite(w).all():
        return 0.0, 0.0
    vector_to_parameters(torch.tensor(w, dtype=torch.float32, device=DEV), net.parameters())
    with torch.no_grad():
        pred = net(X).argmax(1)
    return (pred == y).float().mean().item(), (pred[y == 1] == 1).float().mean().item()


def run(data, ops, attack_norm=None, gate=None, collect=None, seed=0):
    torch.manual_seed(seed)
    clients, root, test = data
    net = model()
    w = parameters_to_vector(net.parameters()).detach().double().cpu().numpy()
    h = np.zeros_like(w)
    h[-2], h[-1] = 1.0, -1.0  # output bias: +normal, -pneumonia
    for _ in range(ROUNDS):
        ref = local_update(net, w, *root)
        deltas = [local_update(net, w, X, y) for X, y in clients]
        if attack_norm is not None:
            for i in range(N_MAL):
                deltas[i] = poison(deltas[i], attack_norm, h)
        if collect is not None:
            collect += [d @ d for d in deltas[N_MAL:]]
        w = w + fltrust(deltas, ref, *ops, gate=gate)
    return evaluate(net, w, *test)


def pick_attack_norm(inv_sqrt, a, b):
    t = np.concatenate([np.linspace(-MAX_T, -1.05, 200), np.linspace(1.05, MAX_T, 200)])
    n = a + (t + 1) * (b - a) / 2
    n = n[n > 0]
    return n[np.argmax(np.abs(inv_sqrt(n) * np.sqrt(n)))]


if __name__ == "__main__":
    np.seterr(over="ignore", invalid="ignore")
    data = load()
    print(f"clients' pneumonia share: {[round(float(y.float().mean()), 2) for _, y in data[0]]}")
    honest = []
    acc, rec = run(data, exact_ops(), collect=honest)
    lo, hi = min(honest), max(honest)
    print(f"{'setting':42} {'accuracy':>8} {'pneumonia recall':>16}")
    print(f"{'exact FLTrust, no attack':42} {acc:8.3f} {rec:16.3f}")
    for deg, margin, newton in CONFIGS:
        a, b = lo / margin, hi * margin
        ops = approx_ops(a, b, deg, newton)
        n = pick_attack_norm(ops[0], a, b)
        label = f"deg {deg}, margin {margin}, newton {newton}"
        rows = [("no attack", run(data, ops)), (f"attacked (size {n / b:.2f}x max)", run(data, ops, attack_norm=n)),
                ("exact FLTrust, same attack", run(data, exact_ops(), attack_norm=n)),
                ("attacked + 1-bit gate", run(data, ops, attack_norm=n, gate=(a, b)))]
        for name, (acc, rec) in rows:
            print(f"{label + ': ' + name:42} {acc:8.3f} {rec:16.3f}")
        (_, clean), (_, att), (_, exact), (_, gated) = rows
        assert exact[0] - att[0] > 0.1 and gated[0] > 0.75, f"{label}: attack/gate result changed"
