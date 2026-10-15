"""Gap F: one oversized update breaks the committee's packed scores (threshold.py).

Every client's u.ref and u.u share one CKKS ciphertext, so one decryption gives all the scores. A client
that sends a huge update (here its honest update x k, so it still points the right way):
  - up to |u|^2 ~ 1e15 decrypts, but CKKS error grows with the largest value in the ciphertext, so the
    other clients' scores get noisy: a lever for flipping honest decisions near a threshold
  - beyond, OpenFHE's decode fails ("approximation error is too high"), the joint decryption fails and
    the round can't finish. One client stops training for everyone, every round, and nobody can tell who.
It never wrapped around to a plausible small norm: OpenFHE's decode check caught every case. An exact
scheme (BFV, or MPC shares mod p) has no such check, so there the same update could wrap silently.
Fix (threshold.Member.read_scores / read_each): if the packed scores fail to decode or hold a norm above
HUGE x median, all members fall back to decrypting each client's scores on their own. The huge update
then breaks only its own score and is rejected. Same 2 scores per client revealed, extra cost only then.
    docker run --rm -v "${PWD}:/work" concilium-research python3 gap_f.py
"""
import os

import numpy as np

import threshold as T
from fl_core import keep, load, local_update
from gap_c import mac

SIZES = (1, 1e4, 1e6, 1e7, 1e8, 1e9, 1e12)


def one_round(cc, pk, shares, keys, root, w, deltas):
    members = [T.Member(cc, pk, s, i == 0, keys, root, w, 0.0, seed=i) for i, s in enumerate(shares)]
    posts = {i: (b, mac(keys[i], 0, b)) for i, b in enumerate(T.encrypt(cc, pk, d) for d in deltas)}
    noises = [m.noise(len(w), len(posts)) for m in members]
    partials = [m.scores(0, posts, noises) for m in members]
    try:  # what the members used to do: fuse the packed scores and use them
        packed = T.fuse(cc, [p[1] for p in partials], T.SLOTS)[::T.BLOCK][:len(deltas)]
    except RuntimeError:
        packed = None
    read = [m.read_scores(partials) for m in members]
    fallback = any(r is None for r in read)
    if fallback:
        each = [m.scores_each() for m in members]
        read = [m.read_each(each) for m in members]
    return packed, fallback, [list(keep(*r)) for r in read]


if __name__ == "__main__":
    clients, root, test = load(non_iid=False)
    cc, pk, shares = T.setup()
    keys = [os.urandom(32) for _ in clients]
    w = np.zeros(test[0].shape[1])
    honest = [local_update(w, X, y) for X, y in clients]
    ref = local_update(w, *root)
    print(f"{'client 0 sends':>15} {'|u0|^2':>9} {'packed: others |u|^2 error':>27} {'fallback':>8} "
          f"{'kept = plaintext':>16} {'client 0 kept':>13}")
    for k in SIZES:
        deltas = [k * honest[0]] + honest[1:]
        norms = np.array([d @ d for d in deltas])
        want = list(keep([d @ ref for d in deltas], norms))
        packed, fallback, kepts = one_round(cc, pk, shares, keys, root, w, deltas)
        err = "decode FAILS" if packed is None else f"{np.abs(packed[1:] - norms[1:]).max():.1e}"
        same = all(kept == want for kept in kepts)
        print(f"{'honest x ' + f'{k:g}':>15} {norms[0]:9.1e} {err:>27} {'yes' if fallback else 'no':>8} "
              f"{'yes' if same else 'NO':>16} {'yes' if 0 in kepts[0] else 'no':>13}", flush=True)
        assert same, "with the fallback, every member must keep exactly what plaintext keep() keeps"
        if k == SIZES[-1]:
            assert packed is None and fallback, "the largest update must break the packed decode (the DoS)"
