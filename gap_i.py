"""Gap I: a participant reposts a transform of another participant's ciphertext (overnight 2026-10-11).

Members only refuse byte-identical duplicates (gap_c.verify_posts). Ciphertexts are malleable, and the
participants hold the public and evaluation keys, so attacker 0 can post, as its own update,
    copy   victim + Enc(0)            a re-randomized copy: new bytes, same update -> victim counted twice
    mask   victim * onehot(k)         victim's coordinate k alone -> the members decrypt u_v[k]^2 and
                                      u_v[k] * ref[k], i.e. the victim's coordinate k, sign included
Members are meant to learn 2 scores per participant; "mask" gives them a coordinate of someone else's update
(useful to a member colluding with a participant; 3 attackers x up to 2 coordinates per round). It needs a
spare multiplicative level: the LR task (31 values, no fold) has one; the CNN's scoring uses all 3, so there
the mask breaks scoring instead (members raise, the run aborts: gap H).
With range proofs (deploy --norm-bound), the attacker must also prove its post: second part below. It can only
prove a vector it knows (its own update, or zeros), and the members' random-projection check then compares the
ciphertext's projections with the proven shares'.
Measured: on LR the range proofs reject every transform. At CNN size (`gap_i.py cnn`) the old tolerance (0.43,
worst-case rounding) let copies through (1/5 at |u| = 0.5, 5/5 at |u| = 0.1).
FIXED 2026-10-11 (owner's pick): range_proof.tolerance uses the rounding's actual spread (0.011 at CNN size),
CHECKS 2 -> 4, and range proofs are on by default (deploy config --norm-bound 1, --no-norm-bound to opt out).
Without range proofs (--no-norm-bound) the first part of this script still holds.
    docker run --rm -e OMP_NUM_THREADS=2 -v "$(pwd -W):/work" -w /work concilium-research python3 gap_i.py [cnn]
(PINE_MALICIOUS isn't needed: the attacker's proofs are honest proofs of vectors it knows.)
"""
import os

import numpy as np
import openfhe as fhe

from fl_core import N_CLIENTS, load, local_update
from gap_c import mac
from threshold import COMMITTEE, Member, chunks, decrypt_std, encrypt, pack, setup

clients, root, test = load(non_iid=False)
dim, victim = test[0].shape[1], 5
cc, pk, shares = setup()
keys = [os.urandom(32) for _ in clients]
w = np.zeros(dim)
deltas = [local_update(w, X, y) for X, y in clients]
ref = local_update(w, *root)
blobs = [encrypt(cc, pk, d) for d in deltas]
v = chunks(blobs[victim], dim)[0]
for name, k in [("copy", None), ("mask", 3), ("mask", 17)]:
    if name == "copy":
        ct = cc.EvalAdd(v, cc.Encrypt(pk, cc.MakeCKKSPackedPlaintext([0.0] * dim)))
    else:
        ct = cc.EvalMult(v, cc.MakeCKKSPackedPlaintext([1.0 if j == k else 0.0 for j in range(dim)]))
    posted = list(blobs)
    posted[0] = pack(fhe.Serialize(ct, fhe.BINARY))
    members = [Member(cc, pk, s, i == 0, keys, root, w, 0.0, seed=i) for i, s in enumerate(shares)]  # fresh round 0
    rnd = 0
    posts = {i: (b, mac(keys[i], rnd, b)) for i, b in enumerate(posted)}
    noises = [m.noise(dim, len(posts)) for m in members]
    partials = [m.scores(rnd, posts, noises) for m in members]
    dots, norms = members[0].read_scores(partials)
    tol = 8 * decrypt_std(COMMITTEE)  # each decrypted score carries the members' flooding
    if name == "copy":
        print(f"copy accepted as participant 0: |u|^2 {norms[0]:.6f} vs victim's {norms[victim]:.6f}, "
              f"u.ref {dots[0]:.6f} vs {dots[victim]:.6f} (flooding std {decrypt_std(COMMITTEE):.1e})")
        assert abs(norms[0] - norms[victim]) < tol and abs(dots[0] - dots[victim]) < tol
    else:
        u = deltas[victim][k]
        print(f"mask k={k}: members read u_k^2 = {norms[0]:+.2e} (true {u * u:.2e}) and u_k ref_k / ref_k = "
              f"{dots[0] / ref[k]:+.4f} (true u_k {u:+.4f}, noise std there {decrypt_std(COMMITTEE) / abs(ref[k]):.1e})")
        assert abs(norms[0] - u * u) < tol and abs(dots[0] - u * ref[k]) < tol
print("gap I reproduced: transformed copies of another participant's ciphertext are accepted and scored" + chr(10))

# with range proofs: the attacker proves what it knows, the ciphertext holds the transform
import hashlib  # noqa: E402
import sys  # noqa: E402

import range_proof as rp  # noqa: E402


def with_proofs(dim, own, victim_u, trials=1, bound=1.0):
    """Attacker proves own (or zeros), posts a transform of the victim's ciphertext. -> list of (case, gap, tol)."""
    p = rp.pine(dim, bound, COMMITTEE)
    verify_key, tol = rp.gen_rand(p.VERIFY_KEY_SIZE), rp.tolerance(COMMITTEE, bound, dim)
    ser = lambda cts: pack(*(fhe.Serialize(c, fhe.BINARY) for c in cts))
    onehot = lambda k: cc.MakeCKKSPackedPlaintext([1.0 if j == k else 0.0 for j in range(min(dim, 8192))])
    members = [Member(cc, pk, s, i == 0, keys, root, np.zeros(dim), 0.0, seed=i, bound=bound) for i, s in enumerate(shares)]
    out = []
    for _ in range(trials):
        vs = chunks(encrypt(cc, pk, victim_u), dim)
        zero = [cc.Encrypt(pk, cc.MakeCKKSPackedPlaintext([0.0])) for _ in vs]
        cases = [("copy, proving its own update", own, ser([cc.EvalAdd(c, z) for c, z in zip(vs, zero)])),
                 ("mask k=3, proving zeros", np.zeros(dim), ser([cc.EvalMult(vs[0], onehot(3))] + vs[1:])
                  if dim <= 8192 else None),  # one chunk only: LR
                 ("honest (control)", own, encrypt(cc, pk, own))]
        for name, proven, blob in cases:
            if blob is None:
                continue
            nonce, public, sh = rp.prove(p, proven)
            started = [rp.prep(p, verify_key, j, nonce, public, sh[j]) for j in range(COMMITTEE)]
            outs = [rp.finish(p, st, [ps for _, ps in started]) for st, _ in started]
            seed = hashlib.sha256(blob + os.urandom(32)).digest()
            rs = np.random.default_rng(int.from_bytes(seed, "big")).choice([-1.0, 1.0], (rp.CHECKS, dim))
            from_ct = members[1].read_projections([m.projections({0: (blob, b"")}, rs) for m in members], 1, rp.CHECKS)[0]
            gap = float(np.max(np.abs(from_ct - rp.combine(p, [rp.project(p, o, rs) for o in outs]))))
            out.append((name, gap, tol))
            print(f"dim {dim:5} {name:30} max |u.r(ct) - u.r(shares)| {gap:.1e} vs tolerance {tol:.1e}: "
                  f"{'scored' if gap <= tol else 'rejected'}", flush=True)
    return out


for name, gap, tol in with_proofs(dim, deltas[0], deltas[victim]):  # LR: the real updates
    assert (gap <= tol) == name.startswith("honest"), "on LR, range proofs must reject the transformed copies"
print("LR: with range proofs, the transformed copies are rejected before scoring" + chr(10))
if sys.argv[1:2] == ["cnn"]:  # CNN size, synthetic updates with honest CNN norms (0.1-1.1, RESEARCH.md)
    rng, d = np.random.default_rng(1), 14146
    u = lambda n: n * rng.normal(0, 1, d) / np.sqrt(d)
    for norm in (0.5, 0.1):
        res = with_proofs(d, u(norm), u(norm), trials=5)
        copies = [g <= t for n, g, t in res if n.startswith("copy")]
        honest = [g <= t for n, g, t in res if n.startswith("honest")]
        print(f"CNN size, |u| = {norm}: copies passing {sum(copies)} of {len(copies)}, honest {sum(honest)} of {len(honest)}")
        assert not any(copies) and all(honest), "at CNN size, range proofs must reject the copies and pass honest posts"
