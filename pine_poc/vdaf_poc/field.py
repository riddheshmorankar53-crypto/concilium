"""Definitions of finite fields used in this spec."""

from typing import Self, TypeVar, cast

from sage.all import GF, PolynomialRing
from sage.rings.finite_rings.finite_field_constructor import FiniteFieldFactory

from vdaf_poc.common import from_le_bytes, to_le_bytes


class Field:
    """The base class for finite fields."""

    # The prime modulus that defines arithmetic in the field.
    MODULUS: int

    # Number of bytes used to encode each field element.
    ENCODED_SIZE: int

    gf: FiniteFieldFactory

    # ppfl: elements hold a plain int (not a Sage/stand-in element) and the arithmetic below skips __init__; ~2x
    # less object work per operation. Same values as the reference.
    def __init__(self, val: int):
        val = int(val)
        assert val < self.MODULUS
        self.val = val % self.MODULUS

    def _new(self, val: int) -> Self:
        out = object.__new__(self.__class__)
        out.val = val
        return out

    @classmethod
    def zeros(cls, length: int) -> list[Self]:
        vec = [cls(cls.gf.zero()) for _ in range(length)]
        return vec

    @classmethod
    def rand_vec(cls, length: int) -> list[Self]:
        """
        Return a random vector of field elements of length `length`.
        """
        vec = [cls(cls.gf.random_element()) for _ in range(length)]
        return vec

    # NOTE: The encode_vec() and decode_vec() methods are excerpted in
    # the document, de-indented, as the figure {{field-derived-methods}}.
    # Their width should be limited to 69 columns after de-indenting, or
    # 73 columns before de-indenting, to avoid warnings from xml2rfc.
    # ===================================================================
    @classmethod
    def encode_vec(cls, vec: list[Self]) -> bytes:
        """
        Encode a vector of field elements `vec` as a byte string.
        """
        # ppfl: one join instead of += per element (quadratic copying: 0.5 s of PINE at CNN size); same bytes
        return b"".join(to_le_bytes(x.as_unsigned(), cls.ENCODED_SIZE) for x in vec)

    @classmethod
    def decode_vec(cls, encoded: bytes) -> list[Self]:
        """
        Parse a vector of field elements from `encoded`.
        """
        L = cls.ENCODED_SIZE
        if len(encoded) % L != 0:
            raise ValueError(
                'input length must be a multiple of the size of an '
                'encoded field element')

        vec = []
        for i in range(0, len(encoded), L):
            encoded_x = encoded[i:i+L]
            x = from_le_bytes(encoded_x)
            if x >= cls.MODULUS:
                raise ValueError('modulus overflow')
            vec.append(cls(x))
        return vec

    # NOTE: The encode_into_bit_vector() and decode_from_bit_vector()
    # methods are excerpted in the document, de-indented, as the figure
    # {{field-bit-rep}}. Their width should be limited to 69 columns
    # after de-indenting, or 73 columns before de-indenting, to avoid
    # warnings from xml2rfc.
    # ===================================================================
    @classmethod
    def encode_into_bit_vector(
            cls,
            val: int,
            bits: int) -> list[Self]:
        """
        Encode the bit representation of `val` with at most `bits` number
        of bits, as a vector of field elements.

        Pre-conditions:

            - `val >= 0`
            - `bits >= 0`
        """
        if val >= 2 ** bits:
            # Sanity check we are able to represent `val` with `bits`
            # number of bits.
            raise ValueError("Number of bits is not enough to represent "
                             "the input integer.")
        encoded = []
        for l in range(bits):
            encoded.append(cls((val >> l) & 1))
        return encoded

    @classmethod
    def decode_from_bit_vector(cls, vec: list[Self]) -> Self:
        """
        Decode the field element from the bit representation, expressed
        as a vector of field elements `vec`.
        """
        bits = len(vec)
        if cls.MODULUS >> bits == 0:
            raise ValueError("Number of bits is too large to be "
                             "represented by field modulus.")
        decoded = cls(0)
        for (l, bit) in enumerate(vec):
            decoded += cls(1 << l) * bit
        return decoded

    def __add__(self, other: Self) -> Self:
        return self._new((self.val + other.val) % self.MODULUS)

    def __neg__(self) -> Self:
        return self._new(-self.val % self.MODULUS)

    def __mul__(self, other: Self) -> Self:
        return self._new(self.val * other.val % self.MODULUS)

    def inv(self) -> Self:
        return self._new(pow(self.val, -1, self.MODULUS))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Field):
            return NotImplemented
        return cast(bool, self.val == other.val)

    def __sub__(self, other: Self) -> Self:
        return self._new((self.val - other.val) % self.MODULUS)

    def __div__(self, other: Self) -> Self:
        return self * other.inv()

    def __pow__(self, n: int) -> Self:
        return self._new(pow(self.val, int(n), self.MODULUS))

    def __str__(self) -> str:
        return str(self.val)

    def __repr__(self) -> str:
        return str(self.val)

    def as_unsigned(self) -> int:
        return self.val


class NttField(Field):
    """
    A field that is suitable for use with the NTT ("number theoretic
    transform") algorithm for efficient polynomial interpolation. Such a field
    defines a large multiplicative subgroup whose order is a power of 2.
    """

    # Order of the multiplicative group generated by `Field.gen()`.
    GEN_ORDER: int

    @classmethod
    def gen(cls) -> Self:
        raise NotImplementedError()


class Field2(Field):
    """The finite field GF(2)."""

    MODULUS = 2
    ENCODED_SIZE = 1

    # Sage finite field object.
    gf = GF(MODULUS)

    def conditional_select(self, inp: bytes) -> bytes:
        """
        Return `inp` unmodified if `self == 1`; otherwise return the all-zero
        string of the same length.

        Implementation note: To protect the code from timing side channels, it
        is important to implement this algorithm in constant time.
        """

        # Convert the element into a bitmask such that `m == 255` if
        # `self == 1` and `m == 0` otherwise.
        m = 0
        v = self.as_unsigned()
        for i in range(8):
            m |= v << i
        return bytes(map(lambda x: m & x, inp))


class Field64(NttField):
    """The finite field GF(2^32 * 4294967295 + 1)."""

    MODULUS = 2**32 * 4294967295 + 1
    GEN_ORDER = 2**32
    ENCODED_SIZE = 8

    # Sage finite field object.
    gf = GF(MODULUS)

    @classmethod
    def gen(cls) -> Self:
        return cls(7)**4294967295


class Field96(NttField):
    """The finite field GF(2^64 * 4294966555 + 1)."""

    MODULUS = 2**64 * 4294966555 + 1
    GEN_ORDER = 2**64
    ENCODED_SIZE = 12

    # Sage finite field object.
    gf = GF(MODULUS)

    @classmethod
    def gen(cls) -> Self:
        return cls(3)**4294966555


class Field128(NttField):
    """The finite field GF(2^66 * 4611686018427387897 + 1)."""

    MODULUS = 2**66 * 4611686018427387897 + 1
    GEN_ORDER = 2**66
    ENCODED_SIZE = 16

    # Sage finite field object.
    gf = GF(MODULUS)

    @classmethod
    def gen(cls) -> Self:
        return cls(7)**4611686018427387897


class Field255(Field):
    """The finite field GF(2^255 - 19)."""

    MODULUS = 2**255 - 19
    ENCODED_SIZE = 32

    # Sage finite field object.
    gf = GF(MODULUS)


##
# POLYNOMIAL ARITHMETIC
#

F = TypeVar("F", bound=Field)


def poly_strip(field: type[F], p: list[F]) -> list[F]:
    """Remove leading zeros from the input polynomial."""
    for i in reversed(range(len(p))):
        if p[i].val != 0:
            return p[:i+1]
    return []


# ppfl: the three helpers below work on plain ints instead of the reference's O(n^2) loops over field objects. Same
# results; `python3 -m vdaf_poc.field` (pine_poc and pine_poc/nosage on sys.path) checks them against the reference.
def poly_mul(field: type[F], p: list[F], q: list[F]) -> list[F]:
    """Multiply two polynomials: Kronecker substitution, i.e. pack each into one big int, let Python's C bigint
    multiply do the convolution, unpack. A coefficient sum is < n * MODULUS^2, so each gets that many bits."""
    if not p or not q:
        return []
    m = field.MODULUS
    width = ((2 * m.bit_length() + min(len(p), len(q)).bit_length()) + 7) // 8
    def pack(v): return int.from_bytes(b"".join(x.val.to_bytes(width, "little") for x in v), "little")
    n = len(p) + len(q) - 1
    b = (pack(p) * pack(q)).to_bytes(n * width, "little")
    mk = field(0)._new
    return poly_strip(field, [mk(int.from_bytes(b[i:i + width], "little") % m) for i in range(0, n * width, width)])


def poly_eval(field: type[F], p: list[F], eval_at: F) -> F:
    """Evaluate a polynomial at a point."""
    m, x, result = field.MODULUS, eval_at.val, 0
    for c in reversed(p):
        result = (result * x + c.val) % m
    return field(result)


def _dft(a: list[int], w: int, m: int) -> list[int]:
    """[sum_j a[j] w^(jk) for k < n], n = len(a) a power of 2 and w of order n: radix-2 NTT, O(n log n)."""
    n = len(a)
    if n == 1:
        return a
    even, odd = _dft(a[0::2], w * w % m, m), _dft(a[1::2], w * w % m, m)
    out, t, h = [0] * n, 1, n // 2
    for k in range(h):
        x = t * odd[k] % m
        out[k], out[k + h] = (even[k] + x) % m, (even[k] - x) % m
        t = t * w % m
    return out


def poly_interp(field: type[F], xs: list[F], ys: list[F]) -> list[F]:
    """Compute the Lagrange interpolation polynomial for the given points.
    The FLP always interpolates at 1, w, w^2, ... (w of order n = len(xs), a power of 2): that is an inverse NTT."""
    n, m = len(xs), field.MODULUS
    w = xs[1].val if n > 1 else 1
    if n & (n - 1) == 0 and all(x.val == pow(w, k, m) for k, x in enumerate(xs)) and pow(w, n, m) == 1             and (n == 1 or pow(w, n // 2, m) != 1):
        n_inv = pow(n, -1, m)
        mk = field(0)._new
        return poly_strip(field, [mk(c * n_inv % m) for c in _dft([y.val for y in ys], pow(w, -1, m), m)])
    R = PolynomialRing(field.gf, 'x')
    p = R.lagrange_polynomial([(x.val, y.val) for (x, y) in zip(xs, ys)])
    return poly_strip(field, list(map(field, p.coefficients())))


if __name__ == "__main__":
    import random
    f = Field64

    def ref_mul(p, q):  # the reference's schoolbook product
        r = [f(0) for _ in range(len(p) + len(q))]
        for i in range(len(p)):
            for j in range(len(q)):
                r[i + j] += p[i] * q[j]
        return poly_strip(f, r)

    for n in (1, 2, 3, 8, 64, 257):
        p, q = f.rand_vec(n), f.rand_vec(random.randint(1, 70))
        p[-1] = f(f.MODULUS - 1)  # largest coefficients: the packing must not overflow
        q[-1] = f(f.MODULUS - 1)
        assert poly_mul(f, p, q) == ref_mul(p, q)
        x = f.rand_vec(1)[0]
        want = f(0)
        for c in reversed(p):
            want = want * x + c
        assert poly_eval(f, p, x) == want
    for n in (1, 2, 4, 32, 256):
        w = f.gen() ** (f.GEN_ORDER // n)
        xs, ys = [w ** k for k in range(n)], f.rand_vec(n)
        c = poly_interp(f, xs, ys)
        assert all(poly_eval(f, c, x) == y for x, y in zip(xs, ys))
        R = PolynomialRing(f.gf, 'x')  # the general path gives the same coefficients
        assert c == poly_strip(f, list(map(f, R.lagrange_polynomial([(x.val, y.val) for x, y in zip(xs, ys)]).coefficients())))
    xs = f.rand_vec(5)  # not roots of unity: the general path
    ys = f.rand_vec(5)
    assert all(poly_eval(f, poly_interp(f, xs, ys), x) == y for x, y in zip(xs, ys))
    assert f(5) * f(5).inv() == f(1) and f(3) - f(5) == -f(2) and f(2) ** -1 == f(2).inv() and f(-1).as_unsigned() == f.MODULUS - 1
    print("ok")
