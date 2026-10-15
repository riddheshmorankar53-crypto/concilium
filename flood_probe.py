"""How big must the committee's decryption noise be? Measures ct.t, the error in the ciphertexts members decrypt,
for threshold.flood_sigma = sqrt(12 tau) 2^(s/2) ct.t (OpenFHE's CKKS noise flooding rule).

Finding (2026-10-02): with OpenFHE's default lazy rescaling, members partially decrypted ciphertexts at scale 2^69
(fresh) to 2^100 (scores, aggregate), where the ciphertext's own error is ~2^55, while each partial adds a fixed
N(0, 2^20): flooding 34 bits too small, not 2^(s/2) larger. Rescaling to 2^50 first (Compress) brings ct.t to
~2^6 for honest-sized updates.
u.u, the one ciphertext x ciphertext product, has an error that grows with |u|^2 beyond |u| ~ 10, so a fixed flood
only holds for bounded updates: participants prove |u| <= B (range_proof.py) and threshold.ct_error(B) is fitted to
the "|u|=B" rows here. Products with a plaintext (u.ref, the clipped aggregate) don't grow with |u|. Errors are decryption minus the exact result over real and imaginary parts (COMPLEX data type), converted to
coefficient units (slot std * scale / sqrt(slots / 2)), and checked by adding a known noise polynomial.
    docker run --rm -v "${PWD}:/work" -w /work concilium-research python3 flood_probe.py
"""
import numpy as np
import openfhe as fhe

from fl_core import load, local_update
from threshold import BLOCK, GROUP, ROTATIONS, SCALE, SLOTS, TOWERS, context, ct_error, flood


def computations(cc, pk, updates, ref):
    """The ciphertexts members decrypt: packed scores (dots, norms) and the aggregate (threshold.Member's ops)."""
    dim = len(ref)
    pt = lambda v: cc.MakeCKKSPackedPlaintext(list(v))
    enc = lambda v: [cc.Encrypt(pk, pt(v[k:k + SLOTS])) for k in range(0, len(v), SLOTS)]
    cts = [enc(u) for u in updates]
    ref_pts = [pt(ref[k:k + SLOTS]) for k in range(0, dim, SLOTS)]
    masks = [pt(np.arange(SLOTS) // BLOCK == j) for j in range(GROUP)]
    heads = pt(np.arange(SLOTS) % BLOCK == 0)

    def into_block(products, j):
        ct = products[0]
        for p in products[1:]:
            ct = cc.EvalAdd(ct, p)
        if dim <= BLOCK:
            return cc.EvalRotate(ct, -BLOCK * j) if j else ct
        for k in [r for r in ROTATIONS if BLOCK <= r < SLOTS]:
            ct = cc.EvalAdd(ct, cc.EvalRotate(ct, k))
        return cc.EvalMult(ct, masks[j])

    def block_sums(ct):
        for k in [r for r in ROTATIONS if 0 < r < BLOCK]:
            ct = cc.EvalAdd(ct, cc.EvalRotate(ct, k))
        return cc.EvalMult(ct, heads)

    dots = norms = None
    for j, c in enumerate(cts):
        d = into_block([cc.EvalMult(x, r) for x, r in zip(c, ref_pts)], j)
        n = into_block([cc.EvalMult(x, x) for x in c], j)
        dots, norms = (d, n) if dots is None else (cc.EvalAdd(dots, d), cc.EvalAdd(norms, n))
    agg = [cc.EvalMult(x, 0.7) for x in cts[0]]  # a clip factor, then the sum and the members' noise ciphertexts
    for c in cts[1:] + [enc(np.zeros(dim))] * 3:
        agg = [cc.EvalAdd(a, x) for a, x in zip(agg, c)]
    return {"fresh": cts[0][0], "dots": block_sums(dots), "norms": block_sums(norms), "aggregate": agg[0]}


def exact(updates, ref):
    dots, norms = np.zeros(SLOTS), np.zeros(SLOTS)
    for j, u in enumerate(updates):
        dots[BLOCK * j], norms[BLOCK * j] = u @ ref, u @ u
    pad = lambda v: np.pad(v[:SLOTS], (0, max(0, SLOTS - len(v))))  # measure over every slot
    return {"fresh": pad(updates[0]), "dots": dots, "norms": norms, "aggregate": pad(0.7 * updates[0] + sum(updates[1:]))}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    clients, _, test = load(non_iid=False)
    lr = [local_update(np.zeros(test[0].shape[1]), X, y) for X, y in clients]
    tasks = {"lr (31 params)": (lr, np.mean(lr, axis=0)),
             "cnn-sized (14146)": ([rng.normal(0, 0.01, 14146) for _ in range(10)], rng.normal(0, 0.01, 14146)),
             "large norms": ([rng.normal(0, 3, 14146) for _ in range(10)], rng.normal(0, 3, 14146))}
    # at the proven bound (range_proof.py): |u| = B exactly, spread over every coordinate or in one spike
    for bound in (1, 10, 100):
        for shape, d in (("spread", 14146), ("spike", 31)):
            ups = []
            for _ in range(10):
                u = rng.normal(0, 1, d) if shape == "spread" else np.eye(d)[rng.integers(d)]
                ups.append(bound * u / np.linalg.norm(u))
            tasks[f"|u|={bound} {shape}"] = (ups, rng.normal(0, 1, d) / np.sqrt(d))
    cc = context()
    kp = cc.KeyGen()
    cc.EvalMultKeyGen(kp.secretKey)
    cc.EvalAtIndexKeyGen(kp.secretKey, ROTATIONS)

    def coeff_log2(values, want, ct):
        v = np.array(values)
        err = np.concatenate([v.real[:len(want)] - want, v.imag[:len(want)]])
        return np.log2(err.std() * ct.GetScalingFactor() / np.sqrt(SLOTS / 2))

    def decrypt(ct):
        pt = cc.Decrypt(kp.secretKey, ct)
        pt.SetLength(SLOTS)
        return pt.GetCKKSPackedValue()

    def share(ct):  # what one member's partial carries, at the ciphertext's own scale
        pt = cc.MultipartyDecryptFusion(cc.MultipartyDecryptLead([ct], kp.secretKey))
        pt.SetLength(SLOTS)
        return pt.GetCKKSPackedValue()

    base = cc.Compress(computations(cc, kp.publicKey, lr, lr[0])["fresh"], TOWERS)
    assert decrypt(base) == decrypt(base), "Decrypt must not add noise of its own, or it can't measure ct.t"
    known = coeff_log2(decrypt(cc.EvalAdd(base, flood(cc, base.GetLevel(), 2.0**30, rng))), exact(lr, lr[0])["fresh"], base)
    print(f"check: a known 2^30 noise polynomial measures as 2^{known:.2f}" + chr(10))
    assert abs(known - 30) < 0.1
    print(f"{'':18} {'ciphertext':10} {'log2 scale':>10} {'ct.t as decrypted':>17} {'ct.t rescaled first':>19}")
    margin = np.inf
    for name, (ups, ref) in tasks.items():
        want = exact(ups, ref)
        # the bound each row must be covered by: none for honest sizes, else the largest |u| in the row
        bound = None if name in ("lr (31 params)", "cnn-sized (14146)") else max(np.linalg.norm(u) for u in ups)
        for kind, ct in computations(cc, kp.publicKey, ups, ref).items():
            low = cc.Compress(ct, TOWERS)
            raw, res = coeff_log2(share(ct), want[kind], ct), coeff_log2(decrypt(low), want[kind], low)
            margin = min(margin, np.log2(ct_error(bound)) - res)
            print(f"{name:18} {kind:10} {np.log2(ct.GetScalingFactor()):10.0f} {raw:17.1f} {res:19.1f}"
                  f"   ct_error 2^{np.log2(ct_error(bound)):.1f}")
    print(chr(10) + f"threshold.ct_error bounds every row, smallest margin {margin:.1f} bits; OpenFHE's fixed share noise 2^20")
    assert margin >= 1, "ct_error must bound the measured error with a bit to spare"
