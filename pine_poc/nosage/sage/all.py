"""Stand-in for the bits of SageMath that PINE's reference code uses (prime fields, Lagrange interpolation), so it
runs in the ppfl image next to OpenFHE. Put pine_poc/nosage on sys.path only where Sage is missing."""
import secrets


class _Elt:
    __slots__ = ("v", "p")

    def __init__(self, v, p):
        self.v, self.p = int(v) % p, p

    def _o(self, x):
        return x.v if isinstance(x, _Elt) else int(x)

    def __add__(self, x): return _Elt(self.v + self._o(x), self.p)
    __radd__ = __add__
    def __sub__(self, x): return _Elt(self.v - self._o(x), self.p)
    def __rsub__(self, x): return _Elt(self._o(x) - self.v, self.p)
    def __mul__(self, x): return _Elt(self.v * self._o(x), self.p)
    __rmul__ = __mul__
    def __neg__(self): return _Elt(-self.v, self.p)
    def __pow__(self, n): return _Elt(pow(self.v, int(n), self.p), self.p)  # n = -1: inverse
    def __eq__(self, x): return isinstance(x, (_Elt, int)) and self.v == self._o(x) % self.p
    def __hash__(self): return hash(self.v)
    def __int__(self): return self.v
    __index__ = __int__
    def __repr__(self): return str(self.v)

    def sqrt(self):
        """Tonelli-Shanks (pine_poc/bench.py's wraparound test)."""
        p, n = self.p, self.v
        assert pow(n, (p - 1) // 2, p) == 1, "not a square"
        q, s = p - 1, 0
        while q % 2 == 0:
            q, s = q // 2, s + 1
        z = next(z for z in range(2, p) if pow(z, (p - 1) // 2, p) == p - 1)
        m, c, t, r = s, pow(z, q, p), pow(n, q, p), pow(n, (q + 1) // 2, p)
        while t != 1:
            i, t2 = 0, t
            while t2 != 1:
                t2, i = t2 * t2 % p, i + 1
            b = pow(c, 1 << (m - i - 1), p)
            m, c, t, r = i, b * b % p, t * b * b % p, r * b % p
        return _Elt(r, p)


class GF:
    def __init__(self, p):
        self.p = p

    def __call__(self, v):
        return _Elt(int(v), self.p)

    def zero(self):
        return _Elt(0, self.p)

    def random_element(self):
        return _Elt(secrets.randbelow(self.p), self.p)


class _Poly:
    def __init__(self, coeffs):
        self.c = coeffs

    def coefficients(self):
        return self.c  # dense, low degree first (Sage's coefficients() skips zeros; the reference means all of them)


class PolynomialRing:
    def __init__(self, gf, name):
        self.p = gf.p

    def lagrange_polynomial(self, points):
        """O(n^2): M(x) = prod (x - xj) once; for each i, M(x) / (x - xi) by synthetic division and M'(xi)."""
        p, n = self.p, len(points)
        xs, ys = [int(x) % p for x, _ in points], [int(y) % p for _, y in points]
        m = [1]  # coefficients, low degree first
        for xj in xs:
            m = [(a - xj * b) % p for a, b in zip([0] + m, m + [0])]
        out = [0] * n
        for xi, yi in zip(xs, ys):
            q, carry = [0] * n, 0  # M / (x - xi), from the top coefficient down
            for k in range(n, 0, -1):
                carry = (m[k] + carry * xi) % p
                q[k - 1] = carry
            den = 0
            for c in reversed(q):  # q(xi) = M'(xi) = prod_{j != i} (xi - xj)
                den = (den * xi + c) % p
            f = yi * pow(den, -1, p) % p
            out = [(o + f * c) % p for o, c in zip(out, q)]
        return _Poly(out)


if __name__ == "__main__":
    F = GF(97)
    assert F(5) ** -1 * F(5) == 1 and int(F(-1)) == 96 and F(3) + 1 == F(4)
    pts = [(F(x), F(x * x + 2 * x + 3)) for x in (1, 2, 5)]
    assert [int(c) for c in PolynomialRing(F, "x").lagrange_polynomial(pts).coefficients()] == [3, 2, 1]
    print("ok")
