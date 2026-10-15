"""PINE parameter variants at CNN size (14146), 3 members: cost and honest/liar verdicts (2026-10-11).
VDAF draft 9.7: circuits with joint randomness MUST use Field128 with >= 1 proof or Field64 with >= 3 proofs;
Pine64 (what range_proof.pine builds) uses Field64 with 2.
    docker run --rm -v "$(pwd -W):/work" -w /work ppfl python3 results/2026-10-11/pine_variants.py 14146"""
import os, sys, time
os.environ["PINE_MALICIOUS"] = "1"
sys.path.insert(0, "/work")
import numpy as np
import range_proof as rp
from vdaf_pine import Pine, Pine64, Pine128
from vdaf_poc.field import Field64
from vdaf_poc.xof import XofTurboShake128
from flp_pine import encode_float
from vdaf_poc.common import gen_rand
dim, M = int(sys.argv[1]), 3
chunk = max(4, int(dim ** 0.5))
b = encode_float(1.0, rp.FRAC)
variants = {
    "Pine64, 2 proofs (old)": lambda: Pine64(b, rp.FRAC, dim, chunk, chunk, M),
    "Pine64, 3+3 proofs": lambda: Pine(Field64, XofTurboShake128, l2_norm_bound=b, num_frac_bits=rp.FRAC, dimension=dim,
        chunk_length=chunk, chunk_length_norm_equality=chunk, num_shares=M, num_proofs=3, num_proofs_norm_equality=3),
    "Pine128, 1 proof (now)": lambda: rp.pine(dim, 1.0, M),
}
rng = np.random.default_rng(0)
u = rp.clip(rng.normal(0, 1, dim), 1.0)
for name, mk in variants.items():
    p = mk()
    vk = gen_rand(p.VERIFY_KEY_SIZE)
    def run(prover, x):
        t0 = time.time(); nonce, pub, shares = rp.prove(prover, x); t1 = time.time()
        st = [rp.prep(p, vk, j, nonce, pub, shares[j]) for j in range(M)]
        out = None if None in st else [rp.finish(p, s, [ps for _, ps in st]) for s, _ in st]
        t2 = time.time()
        ok = out is not None and None not in out
        return ok, t1 - t0, (t2 - t1) / M, sum(map(len, shares)) + len(pub)
    ok, tp, tv, size = run(p, u)
    bad, *_ = run(rp.liar(mk()), 3 * u)
    print(f"{name:24} honest ok {ok}, liar accepted {bad}, prove {tp:.1f}s, verify per member {tv:.1f}s, proof {size/1e6:.2f} MB", flush=True)
