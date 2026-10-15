"""Phase 1: encrypted FedAvg + encrypted poisoning check (TenSEAL CKKS).

Base for gaps B (approximation attacks), C (score leakage), D (key recovery).
    pip install tenseal numpy scikit-learn
    python fl_core.py
"""
import time

import numpy as np
try:  # the research scripts; the deployed tool (Dockerfile target tool) runs without TenSEAL
    import tenseal as ts
except ImportError:
    ts = None
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

N_CLIENTS, N_MAL, ROUNDS, ROOT_SIZE = 10, 3, 20, 20


def load(non_iid, seed=0):
    rng = np.random.default_rng(seed)
    X, y = load_breast_cancer(return_X_y=True)
    X = np.hstack([StandardScaler().fit_transform(X), np.ones((len(X), 1))])  # + bias column
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    root = rng.choice(len(Xtr), ROOT_SIZE, replace=False)  # server's small clean set (FLTrust-style)
    rest = np.setdiff1d(np.arange(len(Xtr)), root)
    # non-IID: sort by label so most clients see mostly one class
    order = rest[np.argsort(ytr[rest], kind="stable")] if non_iid else rng.permutation(rest)
    clients = [(Xtr[i], ytr[i]) for i in np.array_split(order, N_CLIENTS)]
    return clients, (Xtr[root], ytr[root]), (Xte, yte)


def local_update(w, X, y, lr=0.1, epochs=5):
    w0 = w.copy()
    for _ in range(epochs):
        w = w - lr * X.T @ (1 / (1 + np.exp(-X @ w)) - y) / len(y)
    return w - w0


def accuracy(w, X, y):
    return float(((X @ w > 0) == y).mean())


def keep(dots, norms):
    """{index: scale} for the updates to average, from the decrypted scores u.ref and u.u: keep updates
    that point the same way as the server's reference (u.ref > 0), each clipped to the median length.
    With fewer than half the clients malicious the median lies between two honest lengths, so the
    attackers carry at most N_MAL median-lengths of the step. Rejecting updates longer than 3|ref|
    instead dropped honest non-IID updates whenever ref was short, which skewed the kept set toward
    one class and collapsed the CNN (gap_e.py). The scale is computed on decrypted scalars and applied
    as a plaintext factor, so there's nothing to approximate under CKKS."""
    # decrypted norms carry the committee's flooding noise (threshold.py), so a tiny |u|^2 can read <= 0:
    # treat it as tiny (scale 1), not NaN
    med, norms = max(float(np.median(norms)), 0.0), np.maximum(norms, 1e-300)
    # a factor rounded down to 0 drops the update; kept, it would still count in the average's denominator
    # ponytail: rounded to 2^-10 because threshold members fuse the scores independently (~1e-12 apart) and
    # must multiply by the identical factor; fails only within 1e-12 of a step, the same odds as a keep/reject tie
    scales = {i: min(1.0, np.floor(np.sqrt(med / n) * 1024) / 1024) for i, (d, n) in enumerate(zip(dots, norms)) if d > 0}
    return {i: s for i, s in scales.items() if s > 0}


def make_keys():
    """Key holder's context (has secret key) and the server's copy (public + eval keys only)."""
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    ctx.global_scale = 2**40
    ctx.generate_galois_keys()  # needed for dot products (rotations)
    return ctx, ts.context_from(ctx.serialize(save_secret_key=False))


class KeyHolder:
    """Decrypts only what the server asks for, and counts it: every output is a CKKS decryption-oracle query (gap D)."""

    def __init__(self, ctx):
        self.ctx, self.revealed = ctx, 0

    def decrypt(self, ct):
        out = ts.ckks_vector_from(self.ctx, ct.serialize()).decrypt()
        self.revealed += len(out)
        return np.array(out)


def aggregate_plain(deltas, ref, defend):
    kept = keep([d @ ref for d in deltas], [d @ d for d in deltas]) if defend else dict.fromkeys(range(len(deltas)), 1.0)
    return sum(s * deltas[i] for i, s in kept.items()) / len(kept), list(kept)


def aggregate_encrypted(blobs, ref, defend, server_ctx, keyholder):
    encs = [ts.ckks_vector_from(server_ctx, b) for b in blobs]
    kept = dict.fromkeys(range(len(encs)), 1.0)
    if defend:  # scores computed on ciphertext; key holder reveals 2 scalars per client
        kept = keep([keyholder.decrypt(e.dot(ref.tolist()))[0] for e in encs],
                    [keyholder.decrypt(e.dot(e))[0] for e in encs])
    scaled = [encs[i] * s for i, s in kept.items()]  # plaintext factor, also 1.0, so all share one level
    total = scaled[0]
    for e in scaled[1:]:
        total = total + e
    return keyholder.decrypt(total) / len(kept), list(kept)


def run(clients, root, test, n_mal, defend, encrypt):
    w = np.zeros(test[0].shape[1])
    if encrypt:
        ctx, server_ctx = make_keys()
        keyholder = KeyHolder(ctx)
    stats = {"honest_rejected": 0, "mal_accepted": 0, "server_s": 0.0, "bytes": 0}
    for _ in range(ROUNDS):
        ref = local_update(w, *root)
        deltas = [local_update(w, X, y) for X, y in clients]
        for i in range(n_mal):
            deltas[i] = -10 * deltas[i]  # sign-flip + scale poisoning
        if encrypt:
            blobs = [ts.ckks_vector(ctx, d.tolist()).serialize() for d in deltas]  # client side
            stats["bytes"] = len(blobs[0])
            t = time.perf_counter()
            agg, kept = aggregate_encrypted(blobs, ref, defend, server_ctx, keyholder)
            stats["server_s"] += time.perf_counter() - t
        else:
            agg, kept = aggregate_plain(deltas, ref, defend)
        stats["honest_rejected"] += sum(i not in kept for i in range(n_mal, len(deltas)))
        stats["mal_accepted"] += sum(i in kept for i in range(n_mal))
        w = w + agg
    stats["acc"] = accuracy(w, *test)
    stats["revealed"] = keyholder.revealed if encrypt else 0
    return w, stats


def check_server_cannot_decrypt():
    ctx, server_ctx = make_keys()
    ct = ts.ckks_vector_from(server_ctx, ts.ckks_vector(ctx, [1.0, 2.0]).serialize())
    try:
        ct.decrypt()
    except ValueError:
        return True
    return False


if __name__ == "__main__":
    assert check_server_cannot_decrypt(), "server context must not hold the secret key"
    for non_iid in (False, True):
        data = load(non_iid)
        print(f"\n=== {'non-IID' if non_iid else 'IID'} | {N_CLIENTS} clients, {ROUNDS} rounds ===")
        print(f"{'setting':34} {'acc':>6} {'honest rej':>10} {'mal acc':>8} {'server s/rd':>11} {'KB/client':>9} {'revealed':>8}")
        results = {}
        for name, n_mal, defend, enc in [
            ("plaintext, no attack", 0, False, False),
            ("encrypted, no attack", 0, False, True),
            ("encrypted, attack, no defense", N_MAL, False, True),
            ("plaintext, attack, defense", N_MAL, True, False),
            ("encrypted, attack, defense", N_MAL, True, True),
        ]:
            w, s = run(*data, n_mal, defend, enc)
            results[name] = w
            print(f"{name:34} {s['acc']:6.3f} {s['honest_rejected']:10} {s['mal_accepted']:8} "
                  f"{s['server_s'] / ROUNDS:11.3f} {s['bytes'] / 1024:9.0f} {s['revealed']:8}")
        # CKKS must not change the model: encrypted == plaintext up to approximation error
        for a, b in [("plaintext, no attack", "encrypted, no attack"),
                     ("plaintext, attack, defense", "encrypted, attack, defense")]:
            err = np.abs(results[a] - results[b]).max()
            assert err < 1e-3, f"{b} drifted from {a}: {err}"
            print(f"max |w_plain - w_enc| ({b}): {err:.2e}")
