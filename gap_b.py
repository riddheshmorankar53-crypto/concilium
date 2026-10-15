"""Gap B: attacking the polynomial approximations inside an encrypted defense.

Encrypted FLTrust can't compute 1/|u| or ReLU under CKKS, so it uses Chebyshev polynomials
fitted on the norm range the designer expects from honest clients (optionally refined with
Newton steps, the standard FHE trick). Exact FLTrust rescales every update to |ref|, so update
size can't be abused. The approximation is only right inside its range: an attacker picks a
norm outside it. Attack search runs in plaintext (CKKS adds ~1e-9 error, measured in
fl_core.py); every chosen attack point is then re-evaluated under real CKKS.
    python gap_b.py
"""
import numpy as np
try:  # the research scripts; the deployed tool (Dockerfile target tool) runs without TenSEAL
    import tenseal as ts
except ImportError:
    ts = None

from fl_core import N_MAL, ROUNDS, accuracy, load, local_update

# (polynomial degree, range margin, Newton steps). Margin m: range = [min/m, max*m] of honest norm^2.
CONFIGS = [(7, 2, 0), (15, 2, 0), (31, 2, 0), (7, 2, 2), (15, 2, 2), (15, 10, 2), (15, 10, 3)]
MAX_T = 5.0  # attacker stays within |mapped input| <= 5 so the polynomial is still computable under CKKS


def cheb(f, domain, deg):
    return np.polynomial.Chebyshev.interpolate(f, deg, domain=domain)


def exact_ops():
    return (lambda n: 1 / np.sqrt(n)), (lambda x: max(0.0, x))


def approx_ops(a, b, deg, newton):
    poly = cheb(lambda n: 1 / np.sqrt(n), [a, b], deg)

    def inv_sqrt(n):
        y = poly(n)
        for _ in range(newton):
            y = y * (3 - n * y * y) / 2  # Newton step for 1/sqrt(n): only + and x, so CKKS-friendly
        return y

    inv_sqrt.poly = poly
    return inv_sqrt, cheb(lambda x: np.maximum(0, x), [-1, 1], deg)


def fltrust(deltas, ref, inv_sqrt, relu, gate=None):
    rn = np.sqrt(ref @ ref)
    num, den = np.zeros_like(ref), 0.0
    for u in deltas:
        if gate and not gate[0] <= u @ u <= gate[1]:
            continue                   # mitigation: key holder reveals 1 bit "norm^2 in range"
        s = inv_sqrt(u @ u)            # ~ 1/|u|
        t = relu((u @ ref) * s / rn)   # ~ ReLU(cos(u, ref))
        num += t * rn * s * u          # update rescaled to |ref|
        den += t                       # decrypted: one scalar per round
    return num / den if den > 1e-12 else np.zeros_like(ref)


def poison(g, norm, h=None):
    """Honest direction + a harmful push h (default: bias toward 'malignant'); cos > 0 so it earns trust."""
    if h is None:
        h = np.zeros_like(g)
        h[-1] = -1.0
    h = h - (h @ g) / (g @ g) * g
    d = g / np.linalg.norm(g) + h / np.linalg.norm(h)
    return np.sqrt(norm) * d / np.linalg.norm(d)


def run(clients, root, test, ops, attack_norm=None, collect=None, gate=None):
    w = np.zeros(test[0].shape[1])
    for _ in range(ROUNDS):
        ref = local_update(w, *root)
        deltas = [local_update(w, X, y) for X, y in clients]
        if attack_norm is not None:
            for i in range(N_MAL):
                deltas[i] = poison(deltas[i], attack_norm)
        if collect is not None:
            collect += [d @ d for d in deltas[N_MAL:]]
        w = w + fltrust(deltas, ref, *ops, gate=gate)
    return accuracy(w, *test)


_CTX = None


def ckks_inv_sqrt(inv_sqrt, newton, norm_sq):
    """Same polynomial + Newton steps on an encrypted norm^2 (N=32768 for depth)."""
    global _CTX
    if _CTX is None:
        _CTX = ts.context(ts.SCHEME_TYPE.CKKS, 32768, coeff_mod_bit_sizes=[60] + [40] * 19 + [60])
        _CTX.global_scale = 2**40
    poly = inv_sqrt.poly
    a, b = poly.domain
    n = ts.ckks_vector(_CTX, [norm_sq])
    y = (n * (2 / (b - a)) - (a + b) / (b - a)).polyval(np.polynomial.chebyshev.cheb2poly(poly.coef).tolist())
    for _ in range(newton):
        y = y * ((n * y * y) * -0.5 + 1.5)
    return y.decrypt()[0]


def attack(data, ops, a, b):
    """Attacker knows the public range [a, b]; tries norms just outside it."""
    t_grid = np.concatenate([np.linspace(-MAX_T, -1.05, 40), np.linspace(1.05, MAX_T, 40)])
    cands = [n for n in (a + (t + 1) * (b - a) / 2 for t in t_grid) if n > 0]
    return min(cands, key=lambda n: run(*data, ops, attack_norm=n))


if __name__ == "__main__":
    np.seterr(over="ignore", invalid="ignore")  # attacked models diverge to inf/nan; that's the point
    data = load(non_iid=False)
    honest = []
    acc_exact = run(*data, exact_ops(), collect=honest)
    lo, hi = min(honest), max(honest)
    print(f"Honest norm^2 {lo:.2e}..{hi:.2e}; exact FLTrust, no attack: {acc_exact:.3f}\n")
    print(f"{'deg':>3} {'margin':>6} {'newton':>6} {'err in range':>12} {'no attack':>9} {'attacked':>8} "
          f"{'exact, same':>11} {'1-bit gate':>10} {'size/max':>8} {'amplif.':>8} {'CKKS vs plain':>14}")
    for deg, margin, newton in CONFIGS:
        a, b = lo / margin, hi * margin
        ops = approx_ops(a, b, deg, newton)
        grid = np.linspace(a, b, 4000)
        err = np.abs(ops[0](grid) * np.sqrt(grid) - 1).max()
        best = attack(data, ops, a, b)
        no_att, att = run(*data, ops), run(*data, ops, attack_norm=best)
        exact_att, gated = run(*data, exact_ops(), attack_norm=best), run(*data, ops, attack_norm=best, gate=(a, b))
        want = float(ops[0](best))
        got = ckks_inv_sqrt(ops[0], newton, best)
        rel = abs(got - want) / max(1e-12, abs(want))
        print(f"{deg:3} {margin:6} {newton:6} {err:12.1e} {no_att:9.3f} {att:8.3f} {exact_att:11.3f} {gated:10.3f} "
              f"{best / b:8.2f} {want * np.sqrt(best):8.1e} {rel:14.1e}")
        assert gated > 0.9, "range gate should always restore accuracy"
        if (deg, newton) == (7, 0):
            assert exact_att - att > 0.1 and rel < 1e-3, "baseline attack must work and match CKKS"
