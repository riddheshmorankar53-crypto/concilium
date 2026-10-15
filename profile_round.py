"""Where one threshold round spends its time (in-process, the CNN task: 14,146 parameters = 2 ciphertexts).
Threshold decryptions (partial decrypt + fusion) are timed separately from the homomorphic work around them.
The per-participant fallback is forced so it gets measured, though it only runs under gap F attacks.
    docker run --rm -v "${PWD}:/work" concilium-research python3 profile_round.py          # OMP_NUM_THREADS unset: all cores
    docker run --rm -e OMP_NUM_THREADS=4 -v "${PWD}:/work" concilium-research python3 profile_round.py
"""
import os
import time
from collections import defaultdict

import numpy as np

import threshold as T
from gap_c import mac

spent = defaultdict(float)
calls, call_s = defaultdict(int), defaultdict(float)  # the members' OpenFHE calls
_partial, _fuse = T.Member._partial, T.fuse


class Timed:
    """Stands in for a member's crypto context and times every call."""

    def __init__(self, cc):
        self._cc = cc

    def __getattr__(self, name):
        f = getattr(self._cc, name)

        def timed(*a):
            t = time.perf_counter()
            out = f(*a)
            calls[name] += 1
            call_s[name] += time.perf_counter() - t
            return out
        return timed if callable(f) else f


def timed_partial(self, ct):
    t = time.perf_counter()
    out = _partial(self, ct)
    spent["decrypt"] += time.perf_counter() - t
    return out


def timed_fuse(*args):
    t = time.perf_counter()
    out = _fuse(*args)
    spent["decrypt"] += time.perf_counter() - t
    return out


T.Member._partial, T.fuse = timed_partial, timed_fuse


def phase(name, fn):
    """Run fn; charge its time minus the decryptions inside it to name."""
    before, t = spent["decrypt"], time.perf_counter()
    out = fn()
    dec = spent["decrypt"] - before
    spent[name] += time.perf_counter() - t - dec
    spent[name + " (decrypt)"] += dec
    return out


if __name__ == "__main__":
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count())))
    task = {}
    exec(open("tasks/cnn.py").read(), task)  # the deploy task, without importing it as a module
    t0 = time.perf_counter()
    cc, pk, shares = T.setup()
    keygen = time.perf_counter() - t0
    w = task["init"]()
    dim, n = len(w), 10
    keys = [os.urandom(32) for _ in range(n)]
    members = [T.Member(cc, pk, s, i == 0, keys, task["root_data"](), w, 0.0, seed=i,
                        train=lambda w, X, y, rnd: task["train"](w, X, y, 999)) for i, s in enumerate(shares)]
    for m in members:
        m.cc = Timed(cc)
    deltas = phase("train (10 participants, torch CPU)", lambda: [task["train"](w, *task["participant_data"](c), c) for c in range(n)])
    blobs = phase("encrypt (10 participants)", lambda: [T.encrypt(cc, pk, d) for d in deltas])
    posts = {c: (b, mac(keys[c], 0, b)) for c, b in enumerate(blobs)}
    noises = phase("members' noise encryption", lambda: [m.noise(dim, n) for m in members])
    parts = phase("scoring: ref + u.ref, u.u on ciphertexts", lambda: [m.scores(0, posts, noises) for m in members])
    read = phase("scoring: fuse", lambda: [m.read_scores(parts) for m in members])
    each = phase("fallback (forced): per-participant scores", lambda: [m.scores_each() for m in members])
    phase("fallback (forced): fuse", lambda: [m.read_each(each) for m in members])
    out = phase("aggregate: scale, sum, partial decrypt", lambda: [m.aggregate(*r) for m, r in zip(members, read)])
    phase("update: fuse aggregate", lambda: [m.update([p for p, _ in out], None if m.lead else members[0].w)
                                             for m in sorted(members, key=lambda m: not m.lead)])
    total = sum(spent.values())
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')} ({os.cpu_count()} cpus), "
          f"one CNN round, 10 participants x 2 ciphertexts, 3 members in one process; key generation {keygen:.1f} s")
    for name, s in sorted(spent.items(), key=lambda kv: -kv[1]):
        if name != "decrypt" and s > 0.005:
            print(f"  {name:48} {s:7.2f} s  {s / (total - spent['decrypt']):5.1%}")
    print(f"  {'total':48} {total - spent['decrypt']:7.2f} s")
    print("the members' OpenFHE calls (all phases):")
    for name in sorted(call_s, key=lambda k: -call_s[k])[:6]:
        print(f"  {name:28} {calls[name]:5} calls {call_s[name]:6.2f} s {call_s[name] / calls[name] * 1e3:6.2f} ms/call")
