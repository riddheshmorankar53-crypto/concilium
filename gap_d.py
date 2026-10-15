"""Gap D: key recovery from CKKS decryptions (Li-Micciancio 2021), and noise flooding as the fix.

CKKS decryption returns m + e, where e depends on the secret key. Whoever sees the decrypted
floats and the ciphertext (the server built the aggregate, so it has both) re-encodes the floats
to get the exact integer polynomial c0 + c1*s mod q, then solves a linear system for s. With s it
decrypts every client's individual update, undoing all of fl_core / gap_c / threshold.
Fix: fresh noise added at decryption, much larger than 1/scale. Noise encrypted into the
ciphertext does not help (checked at the end). threshold.py measures OpenFHE's partial-decryption noise.
Assumes the naive decryptor releases all N/2 complex slots at float64. APIs that return only the
real parts of the first `dim` slots (TenSEAL, OpenFHE) break this exact linear version, but that
is lost precision, not a proof: lattice variants of the attack remain (untested here).
TenSEAL/OpenFHE don't expose ciphertext polynomials to Python, so this is textbook CKKS in numpy
(same encode/encrypt/decrypt equations, toy ring size N=64, scale 2^30, q = 2^61 - 1).
    python gap_d.py
"""
import numpy as np

from fl_core import N_CLIENTS, load, local_update

N, SCALE, Q = 64, 2.0**30, 2**61 - 1  # ponytail: toy N, not secure; the attack is linear so N doesn't save you
ROOTS = np.exp(1j * np.pi * (2 * np.arange(N) + 1) / N)
V = np.vander(ROOTS, N, increasing=True)
SIGMAS = (0.0, 0.1 / SCALE, 1e-8, 1e-6, 1e-4)  # flooding std, in slot units (same units as threshold.py)


def centered(a):
    return np.array([(int(x) + Q // 2) % Q - Q // 2 for x in a], dtype=object)


def negacyclic(a):
    """Matrix of 'multiply by a' in Z_q[X]/(X^N + 1)."""
    M = np.empty((N, N), dtype=object)
    for j in range(N):
        col = [0] * N
        for i in range(N):
            col[(i + j) % N] = int(a[i]) if i + j < N else -int(a[i])
        M[:, j] = col
    return M


def mul(a, b):
    return np.array([x % Q for x in negacyclic(a) @ b], dtype=object)


def encode(z):
    z = np.concatenate([z, np.zeros(N // 2 - len(z))])
    full = np.concatenate([z, np.conj(z[::-1])])  # conjugate slots -> real coefficients
    return np.array([int(round(x)) for x in np.linalg.solve(V, full).real * SCALE], dtype=object)


def decode(m):
    return V[: N // 2] @ np.array([float(x) for x in centered(m)]) / SCALE


def small(rng, ternary=False):
    x = rng.integers(-1, 2, N) if ternary else np.round(rng.normal(0, 3.2, N)).astype(int)
    return np.array([int(v) for v in x], dtype=object)


def keygen(rng):
    s, a = small(rng, ternary=True), np.array([int(v) for v in rng.integers(0, Q, N, dtype=np.uint64)], dtype=object)
    return s, ((-mul(a, s) + small(rng)) % Q, a)


def encrypt(pk, z, rng):
    u = small(rng, ternary=True)
    return ((mul(pk[0], u) + small(rng) + encode(z)) % Q, (mul(pk[1], u) + small(rng)) % Q)


def decrypt(s, ct, sigma=0.0, rng=None):
    """All N/2 complex slots at float64, as a naive decryptor releases them."""
    out = decode((ct[0] + mul(ct[1], s)) % Q)
    return out + rng.normal(0, sigma, N // 2) + 1j * rng.normal(0, sigma, N // 2) if sigma else out


def solve_mod(A, b):
    """Gauss-Jordan over Z_q (q prime)."""
    A = np.hstack([A % Q, (b % Q).reshape(-1, 1)]).astype(object)
    for c in range(N):
        p = next(r for r in range(c, N) if A[r, c])
        A[[c, p]] = A[[p, c]]
        A[c] = A[c] * pow(int(A[c, c]), -1, Q) % Q
        for r in range(N):
            if r != c and A[r, c]:
                A[r] = (A[r] - A[r, c] * A[c]) % Q
    return A[:, -1]


def recover_key(ct, released):
    """Attacker: re-encode released floats -> c0 + c1*s exactly -> solve for s."""
    return centered(solve_mod(negacyclic(ct[1]), encode(released) - ct[0]))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    clients, _, test = load(non_iid=False)
    dim = test[0].shape[1]
    deltas = [local_update(np.zeros(dim), X, y) for X, y in clients]
    s, pk = keygen(rng)
    posts = [encrypt(pk, d, rng) for d in deltas]
    agg = posts[0]
    for ct in posts[1:]:  # server's homomorphic sum (what it asks to be decrypted)
        agg = ((agg[0] + ct[0]) % Q, (agg[1] + ct[1]) % Q)
    assert np.abs(decrypt(s, agg).real[:dim] / N_CLIENTS - np.mean(deltas, axis=0)).max() < 1e-6

    print("Server sees the released aggregate (1 round), recovers s, decrypts client 0's post:")
    print(f"{'flooding std':>12} {'agg error':>10} {'key recovered':>13} {'client 0 error':>14}")
    for sigma in SIGMAS:
        released = decrypt(s, agg, sigma, rng)
        s_guess = recover_key(agg, released)
        ok = all(s_guess == s)
        leak = np.abs(decrypt(s_guess, posts[0]).real[:dim] - deltas[0]).max()
        print(f"{sigma:12.1e} {np.abs(released.real[:dim] / N_CLIENTS - np.mean(deltas, axis=0)).max():10.1e} "
              f"{str(ok):>13} {leak:14.1e}")
        if sigma == 0:
            assert ok and leak < 1e-6, "attack must recover the key from an exact decryption"
        if sigma >= 1e-8:
            assert not ok and leak > 1, "flooding must break the linear attack"

    # Noise *encrypted into* the ciphertext (a DP-style mask) is part of the message, and the attack
    # never needs the message: it re-encodes the output of whatever ciphertext was decrypted.
    noisy = encrypt(pk, rng.normal(0, 1e-4, dim), rng)
    ct = ((agg[0] + noisy[0]) % Q, (agg[1] + noisy[1]) % Q)
    ok = all(recover_key(ct, decrypt(s, ct)) == s)
    print(f"\nnoise encrypted into the ciphertext (std 1e-4), exact decryption: key recovered {ok}")
    assert ok, "encrypted noise must not stop key recovery; only noise added at decryption does"
