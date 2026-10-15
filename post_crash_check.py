"""Hostile post bytes that crash OpenFHE (not raise) must not take a member down (deploy.Loader).
A fuzz of fresh ciphertexts with a few bytes changed (same size, so the shape checks pass) found 6 of 60 that
kill or hang the deserializer: SIGFPE, an out-of-memory kill, a hang. CRASH_AT is one of them: 8 header bytes
(a modulus field), identical in every fresh ciphertext under this context.
    docker run --rm -v "$(pwd -W):/work" -w /work concilium-research python3 post_crash_check.py"""
import subprocess
import sys

import numpy as np
import openfhe as fhe

import deploy
from threshold import chunks, context, encrypt, pack, unpack

CRASH_AT, CRASH = 303, bytes.fromhex("2ef1ffffffff0000")


def hostile(post):
    first, *rest = unpack(post)
    return pack(first[:CRASH_AT] + CRASH + first[CRASH_AT + len(CRASH):], *rest)


if __name__ == "__main__":
    dim = 31
    cc = context()
    pk = cc.KeyGen().publicKey
    good, bad = encrypt(cc, pk, np.ones(dim)), hostile(encrypt(cc, pk, np.zeros(dim)))
    x = chunks(good, dim)[0]
    fresh = (x.GetLevel(), x.GetNoiseScaleDeg())
    # the bytes really crash OpenFHE in a process of their own (else this check proves nothing)
    open("/tmp/bad", "wb").write(bad)
    open("/tmp/cc", "wb").write(fhe.Serialize(cc, fhe.BINARY))
    r = subprocess.run([sys.executable, "-c", "import openfhe as f, threshold as t;"
                        "f.DeserializeCryptoContextString(open('/tmp/cc','rb').read(), f.BINARY);"
                        f"t.chunks(open('/tmp/bad','rb').read(), {dim})"], capture_output=True, timeout=120).returncode
    assert r < 0, f"the hostile bytes must kill OpenFHE by a signal, got exit {r}"
    loader = deploy.Loader(fhe.Serialize(cc, fhe.BINARY), dim, fresh)
    got = loader.check([good, bad, good])
    print(f"OpenFHE on the hostile post: signal {-r}; loader verdicts good/hostile/good: {got}")
    assert got == [True, False, True], "the loader must drop the hostile post and keep the others"
    assert loader.check([good]) == [True], "the loader must recover after a crash"
