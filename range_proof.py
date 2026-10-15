"""Bounding u before the committee squares it (the u.u gap in threshold.py's flooding).

u.u is the one ciphertext x ciphertext product the committee decrypts, and its error grows with |u|, so the flooding
(threshold.flood_sigma) only holds if |u| is bounded. Each participant now also secret-shares u to the members with
a PINE proof that |u| <= B (pine_poc/, PINE's reference code; Rothblum et al. 2023). That alone says nothing
about the ciphertext, so the members check the two hold the same u: after all posts are in they draw CHECKS random
+-1 vectors r and compare u.r from the ciphertext (a product with a plaintext: its error doesn't grow with |u|) to
u.r from the shares (field arithmetic). A ciphertext off by d passes one check with probability ~ tol / |d|.
Only participants that pass are scored. Cost: members also see CHECKS random projections of each update.
Every message is bytes (threshold.pack, never pickle), so deploy.py relays them through the board.
    docker run --rm -v "${PWD}:/work" -w /work concilium-research python3 range_proof.py
"""
import math
import os
import sys
import time

if __name__ == "__main__":  # client-side self-checks off, so the members (not the client) must catch the liar
    os.environ["PINE_MALICIOUS"] = "1"
PINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pine_poc")
sys.path[:0] = [PINE] + ([] if any(os.path.isdir(f"{p}/sage") for p in sys.path if p) else [f"{PINE}/nosage"])
import numpy as np  # noqa: E402
from flp_pine import encode_float, range_check  # noqa: E402
from vdaf_pine import Pine128  # noqa: E402
from vdaf_poc.common import gen_rand  # noqa: E402

from threshold import pack, unpack  # noqa: E402

FRAC, CHECKS = 15, 4  # fixed-point bits, random projections (4 since 2026-10-11: gap_i.py)


def pine(dim, bound, members):
    chunk = max(4, math.isqrt(dim))
    # Field128, 1 proof: VDAF 9.7 requires Field128 (>= 1 proof) or Field64 with >= 3 when proofs use joint
    # randomness; Pine64 has 2. Same cost (results/2026-10-11/pine_variants.py)
    return Pine128(encode_float(bound, FRAC), FRAC, dim, chunk, chunk, members)


def clip(u, bound):
    """An honest participant's update, shrunk if needed so it provably fits (with room for fixed-point rounding)."""
    n = np.linalg.norm(u)
    return u * min(1.0, bound * (1 - 2.0**-10) / n) if n else u


def liar(p):
    """A participant whose update is out of bounds and who sends the low bits of its norm checks (pine_bench.py).
    Needs PINE_MALICIOUS=1 set before import (it switches the client's own checks off)."""
    valid = p.valid

    def encode(measurement):
        g = [valid.encode_float_into_field(x) for x in measurement]
        _, v, u = range_check(sum((x ** 2 for x in g), valid.field(0)), valid.field(0), valid.sq_norm_bound)
        mask, n = (1 << valid.num_bits_for_sq_norm) - 1, valid.num_bits_for_sq_norm
        return g + valid.field.encode_into_bit_vector(v.as_unsigned() & mask, n) +             valid.field.encode_into_bit_vector(u.as_unsigned() & mask, n)
    valid.encode_gradient_and_norm = encode
    return p


def prove(p, u):
    """Participant: (nonce, public share, one input share per member), all bytes."""
    nonce = gen_rand(p.NONCE_SIZE)
    (wr, vf), shares = p.shard(list(map(float, u)), nonce, gen_rand(p.RAND_SIZE))
    vec = p.flp.field.encode_vec
    enc = [pack(vec(s[0]), vec(s[1]), s[2], s[3]) if isinstance(s[0], list) else pack(*s) for s in shares]
    return nonce, pack(*wr, *vf), enc


def prep(p, verify_key, j, nonce, public, share):
    """Member j: (state, prep share bytes) from its input share, or None if malformed."""
    try:
        parts, pub = unpack(share), unpack(public)
        dec = p.flp.field.decode_vec
        s = (dec(parts[0]), dec(parts[1]), parts[2], parts[3]) if j == 0 else tuple(parts)
        state, (verifier, wr, vf) = p.prep_init(verify_key, j, None, nonce, (pub[:p.SHARES], pub[p.SHARES:]), s)
        return state, pack(p.flp.field.encode_vec(verifier), wr, vf)
    except (ValueError, IndexError):
        return None


def prep_job(args):
    """prep() in a worker process: deploy.py verifies a round's participants in parallel."""
    dim, bound, members, *rest = args
    return prep(pine(dim, bound, members), *rest)


def finish(p, state, prep_shares):
    """Member: my output share of u (field, fixed point) once every member's prep share is in, or None (reject)."""
    try:
        shares = [(p.flp.field.decode_vec(a), b, c) for a, b, c in map(unpack, prep_shares)]
        return p.prep_next(state, p.prep_shares_to_prep(None, shares))
    except (ValueError, IndexError):
        return None


def project(p, out_share, rs):
    """Member: my share of u.r for each +-1 vector r (field elements as ints; alone they're uniformly random)."""
    f = p.flp.field
    out = []
    for r in rs:
        total = f(0)
        for x, sign in zip(out_share, r):
            total = total + x if sign > 0 else total - x
        out.append(total.as_unsigned())
    return out


def combine(p, member_projections):
    """u.r for each r, from every member's project() output."""
    f = p.flp.field
    return [p.valid.decode_float_from_field(sum((f(m[k]) for m in member_projections), f(0)))
            for k in range(len(member_projections[0]))]


def tolerance(committee, bound, dim):
    from threshold import decrypt_std
    # the flooding + fixed-point rounding. Rounding to the nearest 2^-FRAC is uniform per coordinate, so a +-1
    # projection's rounding error has std sqrt(dim/12) 2^-FRAC. The worst case, dim 2^-FRAC, was 0.43 on the CNN:
    # wide enough for a copy of another participant's ciphertext to pass (gap_i.py). The projections are drawn
    # after every post is in, so a cheater can't line its rounding up with them either.
    return 8 * decrypt_std(committee, bound) + 8 * math.sqrt(dim / 12) * 2.0**-FRAC


if __name__ == "__main__":
    import hashlib

    from fl_core import load, local_update
    from gap_c import mac
    from threshold import COMMITTEE, Member, decrypt_std, encrypt, setup

    BOUND = 1.0  # public L2 bound on an update
    t0 = time.time()
    clients, root, test = load(non_iid=False)
    dim = test[0].shape[1]
    us = [local_update(np.zeros(dim), X, y) for X, y in clients]
    rng = np.random.default_rng(0)
    honest_norm = max(np.linalg.norm(u) for u in us)
    assert honest_norm < BOUND, "the bound must admit honest updates"
    cc, pk, keys = setup()
    macs = [os.urandom(32) for _ in us]
    # 10 participants of which 3 cheat, so 7 remain: below gap_c's default minimum of 8
    members = [Member(cc, pk, s, i == 0, macs, root, np.zeros(dim), 0.0, seed=i, min_clients=7, bound=BOUND)
               for i, s in enumerate(keys)]
    p = pine(dim, BOUND, COMMITTEE)
    verify_key = gen_rand(p.VERIFY_KEY_SIZE)  # members only

    # who sends what: (shares of, proof from, ciphertext of)
    off, big = rng.normal(0, 1, dim), 100 * BOUND * us[9] / np.linalg.norm(us[9])
    cases = {c: ("honest", us[c], p, us[c]) for c in range(7)}
    cases[7] = ("huge ciphertext, valid small shares", us[7], p, 1e6 * us[7])
    cases[8] = ("ciphertext off by |d| = 0.5", us[8], p, us[8] + 0.5 * off / np.linalg.norm(off))
    cases[9] = ("|u| = 100 B, lying proof", big, liar(pine(dim, BOUND, COMMITTEE)), big)

    proofs, posts = {}, {}
    for c, (_, shared, prover, enc) in cases.items():
        proofs[c] = prove(prover, shared)
        blob = encrypt(cc, pk, enc)
        posts[c] = (blob, mac(macs[c], 0, blob))
    t_post = time.time()
    outs = {}
    for c, (nonce, public, shares) in proofs.items():  # each member preps its own share; prep shares are exchanged
        started = [prep(p, verify_key, j, nonce, public, shares[j]) for j in range(COMMITTEE)]
        outs[c] = None if None in started else [finish(p, st, [ps for _, ps in started]) for st, _ in started]
        outs[c] = None if outs[c] is None or None in outs[c] else outs[c]
    t_pine = time.time()

    # projections drawn after every post is in, from all posts plus each member's fresh randomness
    seed = hashlib.sha256(b"".join(b for b, _ in posts.values()) + b"".join(os.urandom(32) for _ in members)).digest()
    rs = np.random.default_rng(int.from_bytes(seed, "big")).choice([-1.0, 1.0], (CHECKS, dim))
    proven = {c: posts[c] for c in posts if outs[c] is not None}
    parts = [m.projections(proven, rs) for m in members]
    from_ct = members[1].read_projections(parts, len(proven), CHECKS)
    tol = tolerance(COMMITTEE, BOUND, dim)
    passed = []
    print(f"bound |u| <= {BOUND} (honest updates up to {honest_norm:.2f}), {CHECKS} projections, tolerance {tol:.1e}" + chr(10))
    print(f"{'participant':38} {'PINE':>8} {'max |u.r(ct) - u.r(shares)|':>28} {'scored':>7}")
    for c, (name, *_) in cases.items():
        if outs[c] is None:
            print(f"{name:38} {'rejected':>8} {'-':>28} {'no':>7}")
            continue
        from_shares = combine(p, [project(p, o, rs) for o in outs[c]])
        gap = np.max(np.abs(from_ct[list(proven).index(c)] - from_shares))
        ok = bool(gap <= tol)  # NaN (didn't decode) fails
        passed += [c] * ok
        print(f"{name:38} {'ok':>8} {gap:28.1e} {'yes' if ok else 'no':>7}")
    assert passed == list(range(7)), f"only the honest participants may be scored, got {passed}"

    # the usual round on the participants that passed: u.u is now computed only for proven-bounded updates
    scored = {c: posts[c] for c in passed}
    noises = [m.noise(dim, len(scored)) for m in members]
    dots, norms = members[0].read_scores([m.scores(0, scored, noises) for m in members])
    want = np.array([us[c] @ us[c] for c in passed])
    assert np.abs(norms - want).max() < 8 * decrypt_std(COMMITTEE, BOUND), "scores of the passing updates must be right"
    size = len(pack(proofs[0][0], proofs[0][1], *proofs[0][2]))
    print(chr(10) + f"scored {len(passed)}: |u|^2 match plaintext within {np.abs(norms - want).max():.1e}")
    print(f"proof post {size / 1024:.0f} KB; PINE verification {(t_pine - t_post) / len(cases):.1f} s per participant, "
          f"all members (pure Python); total {time.time() - t0:.0f} s")
