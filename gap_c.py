"""Gap C: a malicious server abuses the key holder's decryptions to read one client's update.

Attacks (naive key holder decrypts whatever the server sends):
  1. one-hot reference: e.dot(one_hot_j) reveals weight j, d queries -> full update
  2. swap: send one client's ciphertext as "the aggregate"
  3. difference: aggregate(all) - aggregate(all but one)
Fix (single key): the key holder decrypts nothing on request. It takes MAC-signed client
posts, computes scores and the sum itself (same work as recomputing the server's result to
verify it), releases only the new global model + keep/reject, once per round.
Limits: the key holder itself is trusted; threshold keys (phase 5) remove that.
    python gap_c.py
"""
import hashlib
import hmac
import os

import numpy as np
try:  # the research scripts; the deployed tool (Dockerfile target tool) runs without TenSEAL
    import tenseal as ts
except ImportError:
    ts = None

from fl_core import (N_MAL, ROUNDS, KeyHolder, accuracy, keep, load, local_update, make_keys, run)

MIN_CLIENTS = 8  # of 10: server can't isolate a client unless it colludes with MIN_CLIENTS - 2 others


def attacks_on_naive(ctx, server_ctx, deltas):
    kh = KeyHolder(ctx)
    encs = [ts.ckks_vector_from(server_ctx, ts.ckks_vector(ctx, d.tolist()).serialize()) for d in deltas]
    target, d = deltas[0], len(deltas[0])
    one_hot = np.array([kh.decrypt(encs[0].dot(np.eye(d)[j].tolist()))[0] for j in range(d)])
    swap = kh.decrypt(encs[0])
    total, rest = encs[0], encs[1]
    for e in encs[1:]:
        total = total + e
    for e in encs[2:]:
        rest = rest + e
    diff = kh.decrypt(total) - kh.decrypt(rest)
    return {name: float(np.abs(got - target).max()) for name, got in
            [("one-hot reference", one_hot), ("swap aggregate", swap), ("difference", diff)]}


def mac(key, rnd, blob):
    return hmac.new(key, rnd.to_bytes(4, "big") + blob, hashlib.sha256).digest()


def verify_posts(expected_rnd, rnd, posts, keys, min_clients=MIN_CLIENTS):
    """posts: {client_id: (blob, tag)}. Raises PermissionError unless it's a fresh, full, signed round.
    keys[cid]: an HMAC key, or a callable (rnd, blob, tag) -> bool that checks a signature."""
    if rnd != expected_rnd:
        raise PermissionError(f"round {rnd} already served or out of order")
    if len(posts) < min_clients:
        raise PermissionError(f"{len(posts)} clients < {min_clients}")
    if len({blob for blob, _ in posts.values()}) != len(posts):
        raise PermissionError("duplicate ciphertexts")
    for cid, (blob, tag) in posts.items():
        key = keys[cid]
        if not (key(rnd, blob, tag) if callable(key) else hmac.compare_digest(tag, mac(key, rnd, blob))):
            raise PermissionError(f"bad MAC/signature for client {cid}")


class GuardedKeyHolder:
    def __init__(self, ctx, root, client_keys, dim):
        self.ctx, self.root, self.keys = ctx, root, client_keys
        self.w, self.rnd = np.zeros(dim), 0
        self.revealed = 0

    def round(self, rnd, posts):
        """posts: {client_id: (blob, mac)}. Returns (new global w, kept ids)."""
        verify_posts(self.rnd, rnd, posts, self.keys)
        ref = local_update(self.w, *self.root)
        encs = {cid: ts.ckks_vector_from(self.ctx, blob) for cid, (blob, _) in posts.items()}
        # scores stay inside the key holder; the server learns only keep/reject
        ids = list(encs)
        scale = {ids[i]: s for i, s in keep([encs[c].dot(ref.tolist()).decrypt()[0] for c in ids],
                                            [encs[c].dot(encs[c]).decrypt()[0] for c in ids]).items()}
        kept = list(scale)
        if len(kept) < 2:
            raise PermissionError("fewer than 2 updates kept; refusing to reveal a single update")
        total = encs[kept[0]] * scale[kept[0]]
        for cid in kept[1:]:
            total = total + encs[cid] * scale[cid]
        agg = np.array(total.decrypt()) / len(kept)
        self.revealed += len(agg)
        self.w, self.rnd = self.w + agg, self.rnd + 1
        return self.w.copy(), kept


def run_guarded(clients, root, test, n_mal):
    ctx, _ = make_keys()
    keys = [os.urandom(32) for _ in clients]
    kh = GuardedKeyHolder(ctx, root, keys, test[0].shape[1])
    w, honest_rejected = kh.w.copy(), 0
    for rnd in range(ROUNDS):
        deltas = [local_update(w, X, y) for X, y in clients]
        for i in range(n_mal):
            deltas[i] = -10 * deltas[i]
        blobs = [ts.ckks_vector(ctx, d.tolist()).serialize() for d in deltas]
        posts = {i: (b, mac(keys[i], rnd, b)) for i, b in enumerate(blobs)}
        w, kept = kh.round(rnd, posts)
        honest_rejected += sum(i not in kept for i in range(n_mal, len(clients)))
    return w, kh, posts, keys


def refused(fn):
    try:
        fn()
    except PermissionError as e:
        return str(e)
    raise AssertionError("guarded key holder allowed a forbidden request")


if __name__ == "__main__":
    clients, root, test = load(non_iid=False)
    ctx, server_ctx = make_keys()
    deltas = [local_update(np.zeros(test[0].shape[1]), X, y) for X, y in clients]

    print("Attacks on a naive key holder (max error recovering client 0's update):")
    for name, err in attacks_on_naive(ctx, server_ctx, deltas).items():
        print(f"  {name:18} {err:.1e}  -> {'LEAKED' if err < 1e-4 else 'safe'}")
        assert err < 1e-4, "attack should succeed against the naive key holder"

    w, kh, posts, keys = run_guarded(clients, root, test, N_MAL)
    w_plain, _ = run(clients, root, test, N_MAL, defend=True, encrypt=False)
    err = np.abs(w - w_plain).max()
    assert err < 1e-3, err
    print(f"\nGuarded run: accuracy {accuracy(w, *test):.3f} (plaintext defended: {accuracy(w_plain, *test):.3f}), "
          f"max weight diff {err:.1e}, values revealed {kh.revealed} (naive defended run: 1020)")

    b0 = posts[0][0]
    fresh = {i: (b, mac(keys[i], ROUNDS, b)) for i, (b, _) in posts.items()}  # validly signed for next round
    print("\nGuarded key holder refuses:")
    for name, fn in [
        ("replay last round", lambda: kh.round(ROUNDS - 1, posts)),
        ("reuse old posts", lambda: kh.round(ROUNDS, posts)),
        ("too few clients", lambda: kh.round(ROUNDS, {i: fresh[i] for i in range(3)})),
        ("duplicate client 0", lambda: kh.round(ROUNDS, {**fresh, 1: (b0, mac(keys[1], ROUNDS, b0))})),
        ("forged MAC", lambda: kh.round(ROUNDS, {**fresh, 1: (b0 + b"x", b"\0" * 32)})),
    ]:
        print(f"  {name:20} {refused(fn)}")
