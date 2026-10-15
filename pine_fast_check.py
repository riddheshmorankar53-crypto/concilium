"""pine_poc's fast run_wr_checks (the wraparound tests, ~90% of PINE's time) must give exactly the reference
loop's results: prover and members must agree bit for bit. Random gradients and seeds, both fields, dimensions
that aren't multiples of 4, and the vectors a liar sends (values near the modulus).
    docker run --rm -v "$(pwd -W):/work" -w /work concilium-research python3 pine_fast_check.py"""
import os
import time

import range_proof as rp  # noqa: F401 (puts pine_poc on the path)
from flp_pine import PineValid
from vdaf_poc.field import Field64, Field128
from vdaf_poc.xof import XofTurboShake128

for field in (Field64, Field128):
    for dim in (1, 3, 4, 5, 97, 1003):
        valid = PineValid(field, l2_norm_bound=1 << 15, num_frac_bits=15, dimension=dim, chunk_length=4)
        for case in range(3):
            if case == 2:  # near the modulus: what a wraparound attack sends
                grad = [field(field.MODULUS - 1 - int.from_bytes(os.urandom(4), "big")) for _ in range(dim)]
            else:
                grad = [field(int.from_bytes(os.urandom(16), "big") % field.MODULUS) for _ in range(dim)]
            seed = os.urandom(16)
            fast = valid.run_wr_checks(grad, XofTurboShake128(seed, b"dst", b"binder"))
            ref = valid.run_wr_checks_reference(grad, XofTurboShake128(seed, b"dst", b"binder"))
            assert [x.val for x in fast] == [x.val for x in ref], (field.__name__, dim, case)
valid = PineValid(Field128, l2_norm_bound=1 << 15, num_frac_bits=15, dimension=14146, chunk_length=118)
grad = [Field128(int.from_bytes(os.urandom(16), "big") % Field128.MODULUS) for _ in range(14146)]
t = time.time()
fast = valid.run_wr_checks(grad, XofTurboShake128(b"s" * 16, b"d", b"b"))
t_fast = time.time() - t
t = time.time()
ref = valid.run_wr_checks_reference(grad, XofTurboShake128(b"s" * 16, b"d", b"b"))
t_ref = time.time() - t
assert [x.val for x in fast] == [x.val for x in ref]
print(f"fast run_wr_checks matches the reference (2 fields, 6 dims, 3 cases each; CNN size): "
      f"{t_ref:.2f} s -> {t_fast:.2f} s at dim 14146")
