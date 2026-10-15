"""Threshold keys (OpenFHE CKKS): no single party can decrypt. Removes gap_c's trusted key holder.

A 3-hospital committee generates the key jointly; each member holds one share. 10 clients encrypt
with the joint public key; the server only relays. Per round:
  1. each member posts an encryption of its own Gaussian noise (hides the aggregate, DP-style).
     This does NOT stop key recovery: it's part of the message, and gap_d.py shows the attack never
     needs the message. Against gap D what counts is the fresh noise OpenFHE adds inside every
     partial decryption; the run measures it (repeat decryptions differ). Fixed size, so heuristic:
     OpenFHE's formal NOISE_FLOODING_MULTIPARTY mode is BGV/BFV only.
  2. range check: each member checks the signed client posts, computes u.ref and u.u for each
     client on the ciphertexts (ref from its own copy of the model and the root set, so the server
     can't choose it), and returns partial decryptions of those scores
  3. each member fuses everyone's score partials itself. If one huge update breaks the packed decode
     (or drowns the other scores' precision, gap_f.py), all members fall back to scoring each client
     on its own ciphertext, so the huge update breaks only its own score and is rejected. Then it keeps
     the clients that pass
     fl_core.keep (the comparison runs on decrypted scalars, so there's no polynomial to attack,
     unlike gap B), rebuilds sum(scale x kept) + sum(noises) (scale clips to the median length), and returns a partial decryption. It refuses
     if its own noise is missing, so one honest member suffices.
  4. every member fuses the 3 aggregate partials and updates its model. Fewer than 3 -> garbage.
Cost: the two scores per client are revealed (same as fl_core), not just 1 bit.
Layout: an update is split into SLOTS-sized ciphertexts. For scoring, each client's per-chunk products
are summed, folded down to one BLOCK (if longer), and moved to block j; one block-sum then gives every
client's score, so step 2 is 2 joint decryptions per round (per GROUP = 16 clients), not 2 per client.
    docker build -t concilium-research .
    docker run --rm -v "${PWD}:/work" concilium-research python3 threshold.py
"""
import json
import os

import numpy as np
import openfhe as fhe

from fl_core import N_MAL, ROUNDS, accuracy, keep, load, local_update, run as run_plain
from gap_c import MIN_CLIENTS, mac, verify_posts

COMMITTEE, SCALE = 3, 2.0**50
SLOTS, BLOCK = 8192, 512  # a ciphertext holds SLOTS values; client j is scored in block j % GROUP of group j // GROUP
GROUP = SLOTS // BLOCK  # clients per packed score ciphertext; more clients -> more groups (2 decryptions each)
ROTATIONS = ([-BLOCK * j for j in range(1, SLOTS // BLOCK)]  # move client j into its block
             + [SLOTS >> k for k in range(1, int(np.log2(SLOTS // BLOCK)) + 1)]  # fold a chunk down to one block
             + [2**k for k in range(int(np.log2(BLOCK)))])  # rotate-and-add: slot BLOCK*j = sum of block j
SIGMAS = (0.0, 1e-3, 1e-1)  # std of DP noise on the global update
HUGE = 1e6  # a packed |u|^2 this many times the median: others' scores lose precision (gap_f.py), fall back
# Noise flooding at partial decryption (gap D; threshold CKKS: Checri et al. Crypto'24). OpenFHE adds a fixed
# N(0, 2^20) to each CKKS share (its NOISE_FLOODING_MULTIPARTY mode is BGV/BFV only) and decrypts wherever the
# ciphertext is: with lazy rescaling at scale 2^100, where its own error is ~2^55 (flood_probe.py). So each member
# rescales to TOWERS towers (scale 2^50, ct.t ~2^6, headroom 2^60 above the scale) and adds its own
# sigma = sqrt(12 tau) 2^(s/2) ct.t, OpenFHE's rule for single-key CKKS. Every member adds the full sigma, so one
# honest member's noise is enough. ponytail: ct.t bound fitted in flood_probe.py; re-measure if the circuit changes.
TOWERS, FLOOD_BITS, FLOOD_QUERIES = 2, 30, 2**16  # s, tau (decryptions per key)


def ct_error(bound=None):
    """Bound on ct.t (flood_probe.py, >= 1 bit of margin): 2^7 covers every decrypted ciphertext while |u| <= 22;
    beyond that u.u's error grows with |u|^2. bound: the L2 bound participants prove (range_proof.py). None: no
    proof, sized for honest updates only, and a huge u can outgrow it."""
    return 2.0**7 * max(1.0, (bound or 0) ** 2 / 2**9)


def flood_sigma(bound=None):
    """Each member's flooding std in coefficient units (~2^31.8 for |u| <= 22)."""
    return np.sqrt(12 * FLOOD_QUERIES) * 2 ** (FLOOD_BITS / 2) * ct_error(bound)


def context():
    p = fhe.CCParamsCKKSRNS()
    p.SetMultiplicativeDepth(3)  # u*u (or u*ref), the fold mask, the final slot mask
    p.SetScalingModSize(int(np.log2(SCALE)))
    p.SetBatchSize(SLOTS)
    p.SetCKKSDataType(fhe.COMPLEX)  # else plaintexts drop imaginary parts and flood() covers half the coefficients
    cc = fhe.GenCryptoContext(p)
    for feature in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.MULTIPARTY):
        cc.Enable(feature)
    return cc


def setup(committee=COMMITTEE):
    """All members' keygen in one process (deploy.py runs the same steps across containers)."""
    cc = context()
    kps = [cc.KeyGen()]
    for _ in range(committee - 1):
        kps.append(cc.MultipartyKeyGen(kps[-1].publicKey))  # chained: joint key = last public key
    tag = kps[-1].publicKey.GetKeyTag()
    # joint relinearization key (for u*u), each member adds its share (OpenFHE threshold-fhe example)
    base = cc.KeySwitchGen(kps[0].secretKey, kps[0].secretKey)
    joint = base
    for kp in kps[1:]:
        joint = cc.MultiAddEvalKeys(joint, cc.MultiKeySwitchGen(kp.secretKey, kp.secretKey, base),
                                    kp.publicKey.GetKeyTag())
    mult = None
    for kp in kps:
        part = cc.MultiMultEvalKey(kp.secretKey, joint, tag)
        mult = part if mult is None else cc.MultiAddEvalMultKeys(mult, part, part.GetKeyTag())
    cc.InsertEvalMultKey([mult])
    # joint rotation keys (packing clients into blocks + block sums)
    idx = ROTATIONS
    cc.EvalAtIndexKeyGen(kps[0].secretKey, idx)
    base_rot = cc.GetEvalAutomorphismKeyMap(kps[0].secretKey.GetKeyTag())
    rots = base_rot
    for kp in kps[1:]:
        t = kp.publicKey.GetKeyTag()
        rots = cc.MultiAddEvalAutomorphismKeys(rots, cc.MultiEvalAtIndexKeyGen(kp.secretKey, base_rot, idx, t), t)
    cc.InsertEvalAutomorphismKey(rots)
    return cc, kps[-1].publicKey, [kp.secretKey for kp in kps]


def _key_towers(sk):
    """A secret key's RNS towers: (json, [values of tower i], [modulus i]). Linear per tower, so Shamir works on it."""
    j = json.loads(fhe.Serialize(sk, fhe.JSON))
    towers = [t["v"]["ptr_wrapper"]["data"] for t in j["value0"]["ptr_wrapper"]["data"]["s"]["v"]]
    return j, towers, [t["m"]["v"] for t in towers]


def share_key(sk, n, t):
    """Shamir t-of-n shares of a member's key share (OpenFHE's ShareKeys isn't in the Python binding).
    share[j-1] goes to member j-1. Any t members can then stand in for this member, so any t together
    can decrypt everything: t-of-n trades collusion resistance for dropout tolerance."""
    _, towers, mods = _key_towers(sk)
    out = [[] for _ in range(n)]
    for tw, m in zip(towers, mods):
        s = np.array(tw["v"], dtype=object)
        coef = [np.array([int.from_bytes(os.urandom(16), "big") % m for _ in s], dtype=object) for _ in range(t - 1)]
        for x in range(1, n + 1):
            out[x - 1].append(sum((c * x ** (k + 1) for k, c in enumerate(coef)), s) % m)
    return [np.array(o).astype(np.uint64) for o in out]  # (towers, ring dim), every modulus < 2^64


def lagrange(me, survivors, m):
    """Coefficient of member `me`'s share (members are 0-based, points 1-based) interpolating f(0) mod m."""
    num = den = 1
    for k in survivors:
        if k != me:
            num, den = num * (k + 1) % m, den * (k - me) % m
    return num * pow(den, -1, m) % m


def recovery_key(sk, dropped_shares, me, survivors):
    """sk + sum over dropped members of lambda_me * my share of their key: survivors' partials then fuse
    as if the dropped members were there. Nobody ever holds a dropped member's key."""
    j, towers, mods = _key_towers(sk)
    for tw, m, i in zip(towers, mods, range(len(towers))):
        v = np.array(tw["v"], dtype=object)
        for share in dropped_shares:
            v = (v + lagrange(me, survivors, m) * share[i].astype(object)) % m
        tw["v"] = [int(x) for x in v]
    return fhe.DeserializePrivateKeyString(json.dumps(j), fhe.JSON)


def pack(*parts):
    """Length-prefixed framing. Never pickle: posts come through the untrusted server."""
    return b"".join(len(p).to_bytes(8, "big") + p for p in parts)


def unpack(data):
    parts, i = [], 0
    while i < len(data):
        n = int.from_bytes(data[i:i + 8], "big")
        parts.append(data[i + 8:i + 8 + n])
        i += 8 + n
    return parts


def n_chunks(dim):
    return -(-dim // SLOTS)


def encrypt(cc, pk, values):
    """One ciphertext per SLOTS values, packed into one blob."""
    return pack(*(fhe.Serialize(cc.Encrypt(pk, cc.MakeCKKSPackedPlaintext(list(values[k:k + SLOTS]))), fhe.BINARY)
                  for k in range(0, len(values), SLOTS)))


def chunks(blob, dim):
    parts = unpack(blob)
    if len(parts) != n_chunks(dim):
        raise PermissionError(f"expected {n_chunks(dim)} ciphertexts, got {len(parts)}")
    return [fhe.DeserializeCiphertextString(b, fhe.BINARY) for b in parts]


def flood(cc, level, sigma, rng):
    """A plaintext whose coefficients are iid N(0, sigma^2): complex Gaussian slots. The slot <-> coefficient map is
    unitary up to sqrt(N/2), so iid slots give iid coefficients; real slots would only cover half of them."""
    n = cc.GetRingDimension() // 2
    std = sigma / SCALE * np.sqrt(n / 2)
    return cc.MakeCKKSPackedPlaintext(list(rng.normal(0, std, n) + 1j * rng.normal(0, std, n)), 1, level)


def decrypt_std(committee, bound=None):
    """Std of each fused value (real part): every member's flooding (~2.1e-4 each for |u| <= 22)."""
    return flood_sigma(bound) * np.sqrt(SLOTS / 2) / SCALE * np.sqrt(committee)


def fuse(cc, partials, dim):
    """Real values are the data; the imaginary parts are only noise. A value that wrapped around the modulus
    (gap F) is garbage in both, so check them, as OpenFHE's REAL decoder does (fails below 5 bits of precision)."""
    pt = cc.MultipartyDecryptFusion(partials)
    pt.SetLength(dim)
    v = np.array(pt.GetCKKSPackedValue())
    if np.abs(v.imag).max() > 2.0**-5:
        raise RuntimeError("approximation error is too high")
    return v.real


class Member:
    def __init__(self, cc, pk, share, lead, client_keys, root, w0, sigma, seed, train=None,
                 committee=COMMITTEE, min_clients=MIN_CLIENTS, bound=None):
        """train(w, X, y, rnd) -> update; default fl_core's logistic regression.
        client_keys[c]: an HMAC key, or a callable (rnd, blob, tag) -> bool (deploy.py: signatures)."""
        self.cc, self.pk, self.share, self.own, self.lead = cc, pk, share, share, lead
        self.committee, self.min_clients = committee, min_clients
        self.flood_rng = np.random.default_rng()  # OS entropy, not the seed: nobody may predict my flooding
        self.bound = bound  # proven L2 bound on updates (range_proof.py), sizes the flooding
        self.keys, self.root, self.sigma, self.rng = client_keys, root, sigma, np.random.default_rng(seed)
        self.train = train or (lambda w, X, y, rnd: local_update(w, X, y))
        self.w, self.rnd, self.my_noise, self.pending, self.kept = w0.copy(), 0, None, None, None
        # Every later op on a deserialized ciphertext is slow (adds ~15x); adding it to a local ciphertext
        # rebinds it. All-zero elements, so every member computes bit-identical ciphertexts (fusion needs that).
        e = cc.Encrypt(pk, cc.MakeCKKSPackedPlaintext([0.0]))
        self.zero = cc.EvalSub(e, e)
        self.masks = [cc.MakeCKKSPackedPlaintext(list(np.arange(SLOTS) // BLOCK == j)) for j in range(GROUP)]
        self.block_heads = cc.MakeCKKSPackedPlaintext(list(np.arange(SLOTS) % BLOCK == 0))

    def recover(self, me, shares, live):
        """Members dropped out: shares = my Shamir shares of their keys (share_key), live = who's left
        (>= t of them). From now on my partials also stand in for the dropped; the lowest live one leads."""
        self.share = recovery_key(self.own, shares, me, live)
        self.lead, self.committee = me == min(live), len(live)

    def _load(self, blob):
        return [self.cc.EvalAdd(self.zero, c) for c in chunks(blob, len(self.w))]

    def noise(self, dim, n_clients=1, split=True):
        # split: per-member std so the sum gets total std sigma * n_clients (the *average* update gets sigma if every
        # client is kept), but colluding members know their part. Not split: every member adds the full std, so one
        # honest member's noise is enough (deploy.py). OS entropy (flood_rng), not the seed: noise anyone can
        # recompute hides nothing.
        std = self.sigma * n_clients / (np.sqrt(self.committee) if split else 1)
        self.my_noise = encrypt(self.cc, self.pk, self.flood_rng.normal(0, std, dim) if std else np.zeros(dim))
        return self.my_noise

    def _partial(self, ct):
        """My share of ct's decryption: rescaled first (deterministic, so all shares fit), plus my flooding."""
        ct = self.cc.Compress(ct, TOWERS)
        decrypt = self.cc.MultipartyDecryptLead if self.lead else self.cc.MultipartyDecryptMain
        return self.cc.EvalAdd(decrypt([ct], self.share)[0], flood(self.cc, ct.GetLevel(), flood_sigma(self.bound), self.flood_rng))

    def scores(self, rnd, posts, noises):
        """Step 2: partial decryptions of all clients' u.ref and all u.u, packed: 2 per GROUP clients."""
        verify_posts(self.rnd, rnd, posts, self.keys, self.min_clients)
        if self.my_noise not in noises:
            raise PermissionError("my noise is missing from the ciphertext I'm asked to decrypt")
        cc, dim, ref = self.cc, len(self.w), self.train(self.w, *self.root, rnd)
        cts = {cid: self._load(b) for cid, (b, _) in posts.items()}
        order = list(cts)
        self.pending, self.rnd = (cts, order, list(noises), ref), self.rnd + 1
        ref_pts = self._plain(ref)
        self.each, packed = {}, []
        for j, cid in enumerate(order):
            b = j % GROUP
            d = self._into_block([cc.EvalMult(c, r) for c, r in zip(cts[cid], ref_pts)], b)
            n = self._into_block([cc.EvalMult(c, c) for c in cts[cid]], b)
            self.each[cid] = (d, n)
            if b == 0:
                packed += [d, n]
            else:
                packed[-2:] = [cc.EvalAdd(packed[-2], d), cc.EvalAdd(packed[-1], n)]
        return [self._partial(self._block_sums(ct)) for ct in packed]  # dots, norms of group 0, then group 1...

    def skip(self, rnd):
        """Nothing is scored or decrypted this round (deploy.py: too few proven updates); the model stays."""
        if rnd != self.rnd:
            raise PermissionError(f"round {rnd} already served or out of order")
        self.rnd, self.kept, self.pending = self.rnd + 1, [], None

    def abandon(self, rnd):
        """The round was cut short (deploy.py: a member was ruled gone mid-round), maybe after scoring: nothing
        more is decrypted, the model stays, the next round is rnd + 1."""
        if rnd not in (self.rnd, self.rnd - 1):
            raise PermissionError(f"round {rnd} isn't the current one")
        self.rnd, self.kept, self.pending = rnd + 1, [], None

    def _plain(self, v):
        return [self.cc.MakeCKKSPackedPlaintext(list(v[k:k + SLOTS])) for k in range(0, len(self.w), SLOTS)]

    def _into_block(self, products, j):
        """Sum a client's per-chunk products; fold to one block if longer; move it to block j."""
        cc, ct = self.cc, products[0]
        for p in products[1:]:
            ct = cc.EvalAdd(ct, p)
        if len(self.w) <= BLOCK:
            return cc.EvalRotate(ct, -BLOCK * j) if j else ct
        for k in [r for r in ROTATIONS if BLOCK <= r < SLOTS]:
            ct = cc.EvalAdd(ct, cc.EvalRotate(ct, k))  # every slot i: sum of slots = i mod BLOCK
        return cc.EvalMult(ct, self.masks[j])  # every block holds the sums, so masking block j moves it there

    def projections(self, posts, rs):
        """range_proof.py: partial decryptions of u.r for each client and each random +-1 vector r, packed like
        the scores (len(rs) per GROUP clients). Only products with plaintexts, so a huge u can't inflate the error
        past the flooding, and it runs before u.u is ever computed."""
        cc, r_pts, groups = self.cc, [self._plain(r) for r in rs], []
        for j, (cid, (blob, _)) in enumerate(posts.items()):
            cts = self._load(blob)
            ps = [self._into_block([cc.EvalMult(c, p) for c, p in zip(cts, pts)], j % GROUP) for pts in r_pts]
            if j % GROUP == 0:
                groups.append(ps)
            else:
                groups[-1] = [cc.EvalAdd(a, p) for a, p in zip(groups[-1], ps)]
        return [self._partial(self._block_sums(ct)) for g in groups for ct in g]

    def read_projections(self, partials, n, k):
        """Fuse projections() -> array (n clients, k vectors); a client whose values don't decode gets NaN."""
        out = np.full((n, k), np.nan)
        for g in range(-(-n // GROUP)):
            for i in range(k):
                try:
                    vals = fuse(self.cc, [p[g * k + i] for p in partials], SLOTS)[::BLOCK]
                except RuntimeError:  # a garbage ciphertext (wrapped values): its whole group fails, decided below
                    continue
                m = min(GROUP, n - g * GROUP)
                out[g * GROUP:g * GROUP + m, i] = vals[:m]
        return out

    def _block_sums(self, ct):
        for k in [r for r in ROTATIONS if 0 < r < BLOCK]:
            ct = self.cc.EvalAdd(ct, self.cc.EvalRotate(ct, k))
        # keep only slot BLOCK*j: the other slots hold partial sums whose differences leak coordinates
        return self.cc.EvalMult(ct, self.block_heads)

    def read_scores(self, score_partials):
        """Step 3a: fuse the packed scores myself -> (dots, norms), or None if they can't be trusted."""
        n = len(self.pending[1])
        if any(len(p) != 2 * -(-n // GROUP) for p in score_partials):
            raise PermissionError("a member's score partials have the wrong length")
        try:
            fused = [fuse(self.cc, [p[k] for p in score_partials], SLOTS)[::BLOCK] for k in range(len(score_partials[0]))]
        except RuntimeError:  # OpenFHE: "approximation error is too high"
            return None
        dots, norms = np.concatenate(fused[0::2])[:n], np.concatenate(fused[1::2])[:n]
        return None if norms.max() > HUGE * max(np.median(norms), 1e-300) else (dots, norms)

    def scores_each(self):
        """Fallback, if any member got None: partial decryptions of each client's two scores on their own."""
        return [(self._partial(self._block_sums(d)), self._partial(self._block_sums(n))) for d, n in self.each.values()]

    def read_each(self, each_partials):
        """Fuse the fallback scores; a client whose scores don't decode is rejected (u.ref = 0)."""
        dots, norms = [], []
        for j in range(len(self.pending[1])):
            try:
                d, n = (fuse(self.cc, [p[j][i] for p in each_partials], SLOTS)[BLOCK * (j % GROUP)] for i in (0, 1))
            except RuntimeError:
                d, n = 0.0, np.inf
            dots.append(d)
            norms.append(n)
        return np.array(dots), np.array(norms)

    def aggregate(self, dots, norms):
        """Step 3b: keep who passes, partial-decrypt sum(scale x kept) + noises."""
        if self.pending is None:
            raise PermissionError("no scored round pending")
        cts, order, noises, ref = self.pending
        self.pending = None
        scale = {order[i]: s for i, s in keep(dots, norms).items()}
        kept = list(scale)
        self.kept = kept
        if len(kept) < 2:  # a sum of 0 or 1 updates would reveal that update: skip the round
            return [], kept
        dim = len(self.w)
        scaled = [[self.cc.EvalMult(c, scale[cid]) for c in cts[cid]] for cid in kept]  # plaintext factor
        totals = scaled[0]
        for parts in scaled[1:] + [self._load(b) for b in noises]:
            totals = [self.cc.EvalAdd(t, p) for t, p in zip(totals, parts)]
        self.kept = kept
        return [self._partial(t) for t in totals], kept

    def update(self, partials, published=None):
        """Step 4: fuse the aggregate myself; adopt the lead's published model if it matches mine.
        Fusion isn't bit-exact (~1e-14 apart), and members must encode an identical ref next round."""
        if len(self.kept) < 2:  # skipped round: nothing was decrypted
            mine = self.w.copy()
        else:
            agg = np.concatenate([fuse(self.cc, [p[k] for p in partials], SLOTS) for k in range(len(partials[0]))])
            mine = self.w + agg[:len(self.w)] / len(self.kept)
        if published is not None and np.abs(published - mine).max() > 1e-6:
            raise PermissionError("published model doesn't match my own fusion")
        self.w = mine if published is None else published.copy()
        return self.w


def run_threshold(clients, root, test, sigma, n_mal):
    cc, pk, shares = setup()
    dim = test[0].shape[1]
    keys = [os.urandom(32) for _ in clients]
    members = [Member(cc, pk, s, i == 0, keys, root, np.zeros(dim), sigma, seed=i) for i, s in enumerate(shares)]
    w, checks, honest_rejected = np.zeros(dim), {}, 0
    for rnd in range(ROUNDS):
        deltas = [local_update(w, X, y) for X, y in clients]
        for i in range(n_mal):
            deltas[i] = -10 * deltas[i]  # same poisoning as fl_core
        blobs = [encrypt(cc, pk, d) for d in deltas]
        posts = {i: (b, mac(keys[i], rnd, b)) for i, b in enumerate(blobs)}
        noises = [m.noise(dim, len(posts)) for m in members]
        score_partials = [m.scores(rnd, posts, noises) for m in members]
        read = [m.read_scores(score_partials) for m in members]
        if any(r is None for r in read):  # one member's doubt is enough: they must all score the same way
            each = [m.scores_each() for m in members]
            read = [m.read_each(each) for m in members]
        out = [m.aggregate(*r) for m, r in zip(members, read)]
        partials, kept = [p for p, _ in out], out[0][1]
        assert all(k == kept for _, k in out), "members must agree on who is kept"
        honest_rejected += sum(i not in kept for i in range(n_mal, len(clients)))
        if rnd == 0:  # decrypting the same ciphertext twice must differ by the flooding (gap D), over all slots
            ct = chunks(noises[0], dim)[0]
            a, b = (fuse(cc, [m._partial(ct) for m in members], SLOTS) for _ in range(2))
            checks["partial decryption noise"] = float((a - b).std() / np.sqrt(2))
        if rnd == 0:  # 2 of 3 shares must not be enough
            try:
                err = np.abs(fuse(cc, [p[0] for p in partials[:2]], dim) / len(kept) - np.mean([deltas[i] for i in kept], axis=0)).max()
                checks["2 of 3 shares"] = f"max error {err:.1e}"
                assert err > 1, "2 of 3 shares must not decrypt"
            except RuntimeError as e:
                checks["2 of 3 shares"] = f"decode fails ({str(e).split(':')[-1].strip()})"
        w = members[0].update(partials)
        for m in members[1:]:
            m.update(partials, w)
    for name, fn in [("replay refused", lambda: members[1].scores(ROUNDS - 1, posts, noises)),
                     ("aggregate without scoring refused", lambda: members[1].aggregate(*read[1]))]:
        try:
            fn()
        except PermissionError as e:
            checks[name] = str(e)
    checks["honest rejected"] = honest_rejected
    return w, checks


def check_groups(n=20, committee=4, dim=9000):
    """More than GROUP participants (2 score groups) and a 4-member committee, 2 chunks per update:
    packed and fallback scores must match plaintext u.ref and u.u."""
    rng = np.random.default_rng(0)
    cc, pk, shares = setup(committee)
    ref, us = rng.normal(0, 1, dim), rng.normal(0, 1, (n, dim)) * rng.uniform(0.5, 2, (n, 1))
    keys = [os.urandom(32) for _ in range(n)]
    members = [Member(cc, pk, s, i == 0, keys, (), np.zeros(dim), 0.0, seed=i, train=lambda *a: ref,
                      committee=committee) for i, s in enumerate(shares)]
    posts = {c: (b, mac(keys[c], 0, b)) for c in range(n) for b in [encrypt(cc, pk, us[c])]}
    noises = [m.noise(dim, n) for m in members]
    parts = [m.scores(0, posts, noises) for m in members]
    want = np.array([us @ ref, (us * us).sum(1)])
    for got in (members[1].read_scores(parts), members[1].read_each([m.scores_each() for m in members])):
        assert np.abs(np.array(got) - want).max() < 8 * decrypt_std(committee), "scores must match plaintext"
    return len(parts[0]) // 2


def check_dropout(committee=3, t=2, dim=100):
    """The lead drops out: with t-of-n shares the other two still score and aggregate, one alone can't."""
    rng = np.random.default_rng(1)
    cc, pk, keys = setup(committee)
    shares = [share_key(k, committee, t) for k in keys]  # shares[i][j]: member j's share of member i's key
    ref, us = rng.normal(0, 1, dim), rng.normal(0, 1, (8, dim))
    macs = [os.urandom(32) for _ in us]
    members = [Member(cc, pk, s, i == 0, macs, (), np.zeros(dim), 0.0, seed=i, train=lambda *a: ref,
                      committee=committee) for i, s in enumerate(keys)]
    live = [1, 2]
    for j in live:
        members[j].recover(j, [shares[0][j]], live)
    members = [members[j] for j in live]
    posts = {c: (b, mac(macs[c], 0, b)) for c in range(len(us)) for b in [encrypt(cc, pk, us[c])]}
    noises = [m.noise(dim, len(us)) for m in members]
    parts = [m.scores(0, posts, noises) for m in members]
    dots, norms = members[0].read_scores(parts)
    assert np.abs(dots - us @ ref).max() < 8 * decrypt_std(len(live)), "2 of 3 must score after a dropout"
    out = [m.aggregate(dots, norms) for m in members]
    for m in members:
        m.update([p for p, _ in out])
    kept, want = out[0][1], np.mean([s * us[c] for c, s in keep(us @ ref, (us * us).sum(1)).items()], axis=0)
    assert np.abs(members[0].w - want).max() < 8 * decrypt_std(len(live)), "2 of 3 must aggregate after a dropout"
    alone = Member(cc, pk, keys[1], True, macs, (), np.zeros(dim), 0.0, seed=9)
    alone.recover(1, [shares[0][1], shares[2][1]], [1])  # t - 1 shares: interpolation is wrong
    ct = chunks(posts[0][0], dim)[0]
    try:
        err = np.abs(fuse(cc, [alone._partial(ct)], dim) - us[0]).max()
    except RuntimeError:
        err = np.inf
    assert err > 1, "1 of 3 must not decrypt"
    return len(kept), err


if __name__ == "__main__":
    kept, err = check_dropout()
    print(f"lead dropped, 2-of-3 shares: 2 members scored and aggregated ({kept} kept), 1 alone: error {err:.1e}\n")
    print(f"{check_groups()} score groups for 20 participants, 4-member committee: scores match plaintext\n")
    clients, root, test = load(non_iid=False)
    w_plain, _ = run_plain(clients, root, test, N_MAL, defend=True, encrypt=False)
    w_naive, _ = run_plain(clients, root, test, N_MAL, defend=False, encrypt=False)
    print(f"{N_MAL} poisoned clients. plaintext FedAvg: {accuracy(w_naive, *test):.3f}, "
          f"plaintext defended: {accuracy(w_plain, *test):.3f}\n")
    print(f"{'noise std':>9} {'accuracy':>8} {'max |w - plain|':>15} {'honest rej':>10}")
    for sigma in SIGMAS:
        w, checks = run_threshold(clients, root, test, sigma, N_MAL)
        err = np.abs(w - w_plain).max()
        print(f"{sigma:9.0e} {accuracy(w, *test):8.3f} {err:15.1e} {checks['honest rejected']:10}")
        if sigma == 0:
            # the flooding is a random walk on the model: <= decrypt_std per round (divided by the kept count)
            assert err < 10 * decrypt_std(COMMITTEE) * np.sqrt(ROUNDS), "threshold defended must match plaintext defended"
            ratio = checks["partial decryption noise"] / decrypt_std(COMMITTEE)
            assert 0.9 < ratio < 1.1, f"decryption noise must be the flooding the members add (got {ratio:.2f}x)"
            assert "replay refused" in checks and "aggregate without scoring refused" in checks
            first = checks
    print(f"\n2 of 3 shares fused: {first['2 of 3 shares']}")
    print(f"decryption noise: std {first['partial decryption noise']:.1e} per value, predicted "
          f"{decrypt_std(COMMITTEE):.1e} (each member floods with 2^{np.log2(flood_sigma()):.1f} in coefficients: "
          f"{FLOOD_BITS}-bit statistical security for {FLOOD_QUERIES} decryptions, ct.t <= 2^{np.log2(ct_error()):.0f})")
    print(f"replayed round: refused ({first['replay refused']})")
    print(f"aggregate without scoring: refused ({first['aggregate without scoring refused']})")
