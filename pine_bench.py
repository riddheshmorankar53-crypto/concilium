"""PINE's reference code with 3 aggregators: honest updates aggregate; the gap F wraparound and an oversized
update that lies about its norm are rejected by the aggregators. Timing and sizes vs dimension.

The reference code (Python on SageMath, pinned commits) is research code, so timings are only relative.
The client-side checks that stop an honest client from sending a bad update are switched off
(PINE_MALICIOUS=1), so the aggregators, not the client, have to catch it.
    python pine_bench.py fetch       # on the host: downloads the POC + vdaf_poc (draft-11) into pine_poc/
    docker run --rm -v "${PWD}/pine_poc:/pine" -w /pine -e PINE_MALICIOUS=1 sagemath/sagemath:latest sage -python bench.py 100 1000
"""
import io
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

PINE = "https://raw.githubusercontent.com/junyechen1996/draft-chen-cfrg-vdaf-pine/4d11ac826606cf8b08e4b34f815ad14ba7397593/poc/"
VDAF = "https://codeload.github.com/cfrg/draft-irtf-cfrg-vdaf/tar.gz/refs/tags/draft-irtf-cfrg-vdaf-11"
# a malicious client: claim every wraparound check passed and send whatever low bits fit
PATCHES = [
    ("import math\n", "import math\nimport os\nMALICIOUS = os.environ.get('PINE_MALICIOUS') == '1'\n"),
    ("            if is_in_range and wr_success_count < self.num_wr_successes:",
     "            if (is_in_range or MALICIOUS) and wr_success_count < self.num_wr_successes:"),
    ("                wr_check_v.as_unsigned(), self.num_bits_for_wr_check,",
     "                wr_check_v.as_unsigned() & ((1 << self.num_bits_for_wr_check) - 1) if MALICIOUS\n"
     "                else wr_check_v.as_unsigned(), self.num_bits_for_wr_check,"),
    ("        if wr_success_count != self.num_wr_successes:",
     "        if wr_success_count != self.num_wr_successes and not MALICIOUS:"),
]

BENCH = '''"""Run inside the SageMath container (see pine_bench.py)."""
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
        return g + valid.field.encode_into_bit_vector(v.as_unsigned() & mask, n) + \\
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
    print(f"  wraparound (x^2 = 2 mod p): {wrap}\\n  |u| = 100 x bound, lying norm bits: {big}", flush=True)
    assert wrap.startswith("rejected") and big.startswith("rejected")
'''


def get(url):
    return urllib.request.urlopen(url).read()


def fetch(dest="pine_poc"):
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest)
    for name in ("vdaf_pine.py", "flp_pine.py", "field.py", "xof_hmac_sha256_aes128.py", "VERSION"):
        text = get(PINE + name).decode()
        if name == "flp_pine.py":
            for old, new in PATCHES:
                assert text.count(old) == 1, old
                text = text.replace(old, new)
        open(os.path.join(dest, name), "w", encoding="utf-8").write(text)
    with tarfile.open(fileobj=io.BytesIO(get(VDAF))) as tar:
        for m in tar.getmembers():
            if "/poc/vdaf_poc/" in m.name and m.isfile():
                m.name = os.path.join("vdaf_poc", m.name.split("/poc/vdaf_poc/")[1])
                tar.extract(m, dest)
    # pycryptodomex for the container's Python 3.12 (pip inside the Sage image didn't install it)
    subprocess.run([sys.executable, "-m", "pip", "download", "-q", "pycryptodomex", "--platform", "manylinux2014_x86_64",
                    "--only-binary=:all:", "--python-version", "3.12", "-d", os.path.join(dest, "wheels")], check=True)
    for whl in os.listdir(os.path.join(dest, "wheels")):
        zipfile.ZipFile(os.path.join(dest, "wheels", whl)).extractall(dest)
    open(os.path.join(dest, "bench.py"), "w", encoding="utf-8").write(BENCH)
    print(f"ready in {dest}/")


if __name__ == "__main__":
    fetch()
