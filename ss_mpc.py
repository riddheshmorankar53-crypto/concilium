"""Prototype of RESEARCH.md "Evaluation: secret sharing with PINE" (Next steps item 1): no HE, the members learn 2 bits.

Participants secret-share their update u among the 3 members with a PINE proof of |u| <= B (range_proof.py,
B = |ref| as the evaluation picked). PINE leaves each member an additive share of u in Field128, fixed point. u.ref
is linear in u, so each member computes its share of it locally (ref is the members' own, fixed point too). The
one nonlinear step, the sign of u.ref, runs in MP-SPDZ (Dockerfile.mpc) with the same prime, so the shares go in
as they are: MASCOT, malicious security with any 2 of the 3 members corrupt. Only the sign bit comes out.
The kept updates' sum would then be the sum of the kept shares (linear, not shown).

Checks (asserted): over 5 FedAvg rounds of the LR task with 3 sign-flip attackers, the MPC bits equal the plaintext
sign of the fixed-point u.ref and the float u.ref, and every attacker is rejected while u.ref < 0.
Then times MASCOT (and semi-honest `semi` for scale), with and without edaBits, per round of 10..100 participants.
    docker build -f Dockerfile.mpc -t concilium-mpc .
    MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/work" concilium-mpc python3 ss_mpc.py
Limits: a member can input a wrong share and flip any participant's bit (MASCOT authenticates what a member
inputs, not that it is its true PINE share): the CKKS version doesn't have this, since there every member computes
on the same public ciphertexts. Fixing it needs shares the participant authenticates (e.g. MACs under a key
shared by the members, checked inside the MPC). All parties on one machine: times exclude network latency.
"""
import os
import re
import subprocess
import sys
import time

import numpy as np

import range_proof as rp
from tasks import lr

MPSPDZ, MEMBERS, LIMB = "/opt/mp-spdz", 3, 43  # shares go in as three 43-bit limbs: plain 64-bit integer inputs
PROGRAM = """
n = int(program.args[1])
program.set_bit_length(64)  # |u.ref| < 2^63 in fixed point: B^2 2^30 with B = |ref| < 2^16
s = [sint.Array(n) for _ in range(3)]
for j in range(3):
    limbs = [sint.Array(n) for _ in range(3)]
    for k in range(3):
        limbs[k].input_from(j)
    @for_range_opt(n)
    def _(c):
        s[j][c] = limbs[0][c] + limbs[1][c] * 2**43 + limbs[2][c] * 2**86
out = cint.Array(n)
@for_range_opt(n)
def _(c):
    out[c] = (s[0][c] + s[1][c] + s[2][c] > 0).reveal()
print_str('bits ')
@for_range(n)
def _(c):
    print_str('%s ', out[c])
print_ln('')
"""


def compile_program(n, prime, flags=()):
    with open(f"{MPSPDZ}/Programs/Source/sign.mpc", "w") as f:
        f.write(PROGRAM)
    subprocess.run(["./compile.py", *flags, "-P", str(prime), "sign", str(n)], cwd=MPSPDZ, check=True,
                   capture_output=True)


def mpc_signs(shares, prime, protocol="mascot", flags=()):
    """shares[j][c]: member j's share of participant c's u.ref (int mod prime). -> (bits, seconds, MB all parties)."""
    n = len(shares[0])
    compile_program(n, prime, flags)
    mask = (1 << LIMB) - 1
    for j in range(MEMBERS):  # input_from reads each Array in turn: all low limbs, then middle, then high
        with open(f"{MPSPDZ}/Player-Data/Input-P{j}-0", "w") as f:
            f.write(" ".join(str((x >> (LIMB * k)) & mask) for k in range(3) for x in shares[j]))
    t = time.time()
    procs = [subprocess.Popen([f"./{protocol}-party.x", str(j), f"sign-{n}", "-N", str(MEMBERS), "-h", "localhost",
                               "-pn", "14000", "-P", str(prime)], cwd=MPSPDZ, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) for j in range(MEMBERS)]
    outs = [p.communicate()[0] for p in procs]
    secs = time.time() - t
    if any(p.returncode for p in procs):
        raise RuntimeError(outs[0][-2000:])
    bits = [int(b) for b in next(ln for ln in outs[0].split("\n") if ln.startswith("bits ")).split()[1:]]
    mb = float(re.search(r"Global data sent = ([\d.e+]+) MB", outs[0]).group(1))
    return bits, secs, mb


def shares_of_dots(p, ref, updates, verify_key):
    """Every member's share of u.ref for each update, through PINE (prove, verify, output shares)."""
    f, enc = p.flp.field, p.valid.encode_float_into_field
    ref_f = [enc(float(x)) for x in ref]
    dots = [[] for _ in range(MEMBERS)]
    for u in updates:
        nonce, public, enc_shares = rp.prove(p, u)
        prepped = [rp.prep(p, verify_key, j, nonce, public, enc_shares[j]) for j in range(MEMBERS)]
        outs = [rp.finish(p, s, [ps for _, ps in prepped]) for s, _ in prepped]
        assert all(o is not None for o in outs), "an honestly clipped update must pass PINE"
        for j, o in enumerate(outs):  # local: a share times a public value, summed
            total = f(0)
            for x, r in zip(o, ref_f):
                total = total + x * r
            dots[j].append(total.as_unsigned())
    return dots


def check(rounds=5, attackers=(0, 1, 2)):
    w = lr.init()
    for rnd in range(rounds):
        ref = lr.train(w, *lr.root_data(), 0)
        bound = float(np.linalg.norm(ref))
        p = rp.pine(len(w), bound, MEMBERS)
        prime = p.flp.field.MODULUS
        raw = [lr.train(w, *lr.participant_data(c), c) for c in range(10)]
        updates = [rp.clip(-u if c in attackers else u, bound) for c, u in enumerate(raw)]
        dots = shares_of_dots(p, ref, updates, os.urandom(p.VERIFY_KEY_SIZE))
        bits, secs, mb = mpc_signs(dots, prime)
        fixed = [int(sum(d[c] for d in dots) % prime) for c in range(10)]  # what the shares add up to
        fixed = [int(0 < x < prime // 2) for x in fixed]
        floats = [int(float(u @ ref) > 0) for u in updates]
        print(f"round {rnd}: |ref| {bound:.3f}, MPC bits {bits}, {secs:.1f} s, {mb:.1f} MB (all members)")
        assert bits == fixed == floats, (bits, fixed, floats)
        assert all(bits[c] == 0 for c in attackers), "a sign-flipped update must point away from ref"
        kept = [u for u, b in zip(updates, bits) if b]
        w = w + np.mean(kept, axis=0)
    print(f"LR after {rounds} rounds with the MPC filter: {lr.evaluate(w)}")


# protocol[:compile flags]; -Y: edaBits for comparisons. MPC_VARIANTS="lowgear,lowgear:-Y" python3 ss_mpc.py ...
VARIANTS = [(v.split(":")[0], tuple(v.split(":")[1:])) for v in
            os.environ.get("MPC_VARIANTS", "mascot,mascot:-Y,semi,semi:-Y").split(",")]


def bench(sizes=(10, 100)):
    prime = rp.pine(31, 1.0, MEMBERS).flp.field.MODULUS
    rng = np.random.default_rng(0)
    print("participants | protocol | seconds | MB sent, all members | per participant")  # cost is linear in n
    for n in sizes:
        dots = [int(x) % prime for x in rng.integers(-2**40, 2**40, n)]  # signed values, shared at random
        shares = [[int.from_bytes(os.urandom(16), "big") % prime for _ in range(n)] for _ in range(2)]
        shares.append([(d - a - b) % prime for d, a, b in zip(dots, *shares)])
        want = [int(0 < d < prime // 2) for d in dots]
        for proto, flags in VARIANTS:
            bits, secs, mb = mpc_signs(shares, prime, proto, flags)
            assert bits == want, f"{proto} {flags} gave wrong signs"
            print(f"{n} | {proto} {' '.join(flags)} | {secs:.1f} | {mb:.1f} | {secs / n * 1000:.0f} ms, "
                  f"{mb / n * 1000:.0f} KB")


if __name__ == "__main__":
    if not os.path.exists(f"{MPSPDZ}/Player-Data/P0.pem"):
        subprocess.run(["Scripts/setup-ssl.sh", str(MEMBERS)], cwd=MPSPDZ, check=True, capture_output=True)
    args = [a for a in sys.argv[1:] if a != "nocheck"]  # [sizes, e.g. 10,100] [nocheck]
    if "nocheck" not in sys.argv:
        check()
    bench(*([tuple(map(int, args[0].split(",")))] if args else []))
