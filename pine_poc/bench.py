"""Run inside the SageMath container (see pine_bench.py)."""
import math, random, sys, time
from sage.all import GF
from vdaf_poc.common import gen_rand
from vdaf_poc.vdaf import run_vdaf
from flp_pine import encode_float, range_check
from vdaf_pine import Pine64

FRAC, BOUND, SHARES = 15, 1.0, 3


def make(d):
    chunk = max(4, math.isqrt(d))
    return Pine64(encode_float(BOUND, FRAC), FRAC, d, chunk, chunk, SHARES)


def honest(d, norm=0.9):
    v = [random.gauss(0, 1) for _ in range(d)]
    s = norm / math.sqrt(sum(x * x for x in v))
    return [x * s for x in v]


def malicious(pine, override):
    """Encode chosen field elements for some coordinates; send the low bits of an out-of-range norm (a lie)."""
    valid = pine.valid

    def encode(measurement):
        g = [valid.encode_float_into_field(x) for x in measurement]
        for i, x in override.items():
            g[i] = valid.field(x)
        _, v, u = range_check(sum((x ** 2 for x in g), valid.field(0)), valid.field(0), valid.sq_norm_bound)
        mask, n = (1 << valid.num_bits_for_sq_norm) - 1, valid.num_bits_for_sq_norm
        return g + valid.field.encode_into_bit_vector(v.as_unsigned() & mask, n) + \
            valid.field.encode_into_bit_vector(u.as_unsigned() & mask, n)
    valid.encode_gradient_and_norm = encode
    return pine


def run(pine, meas):
    try:
        return run_vdaf(pine, gen_rand(pine.VERIFY_KEY_SIZE), None, [gen_rand(pine.NONCE_SIZE) for _ in meas], meas)
    except ValueError as e:
        return f"rejected by the aggregators ({str(e)[:55]})"


p = make(4).valid.field.MODULUS
x = int(GF(p)(2).sqrt())  # x^2 = 2 mod p, but x itself is far outside any bound
print(f"field p = {p}; wraparound coordinate x = {x}, x^2 mod p = {x * x % p}", flush=True)
for d in map(int, sys.argv[1:]):
    meas = [honest(d), honest(d)]
    t = time.time()
    agg = run(make(d), meas)
    err = max(abs(a - b - c) for a, b, c in zip(agg, *meas))
    pine = make(d)
    n = pine.MEAS_LEN + pine.flp.PROOF_LEN * pine.PROOFS + pine.flp_norm_equality.PROOF_LEN * pine.PROOFS_NORM_EQUALITY
    print(f"d={d}: {(time.time() - t) / 2:.0f} s/report, aggregate error {err:.1e}, leader share {n} field elements "
          f"({n * 8 / 1024:.0f} KB, {n / d - 1:.0%} over the raw vector)", flush=True)
    assert err < 1e-3
    wrap, big = run(malicious(make(d), {0: x}), [honest(d)]), run(malicious(make(d), {}), [honest(d, 100 * BOUND)])
    print(f"  wraparound (x^2 = 2 mod p): {wrap}\n  |u| = 100 x bound, lying norm bits: {big}", flush=True)
    assert wrap.startswith("rejected") and big.startswith("rejected")
