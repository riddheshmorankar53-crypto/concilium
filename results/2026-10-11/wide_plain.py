"""Plaintext check before deploying tasks/cnn_wide.py (2026-10-11): accuracy of the defended filter (fl_core.keep,
updates clipped to B = 1 as in deploy) and honest update norms vs B, small vs wide CNN; no attack and sign-flip x10.
    ~/ppfl-gpu/Scripts/python results/2026-10-11/wide_plain.py"""
import importlib
import sys

import numpy as np

sys.path.insert(0, ".")
from fl_core import N_MAL, keep  # noqa: E402


def clip(u, bound):  # range_proof.clip (importing it loads PINE, Linux only)
    n = np.linalg.norm(u)
    return u * min(1.0, bound * (1 - 2.0**-10) / n) if n else u

for name in ("cnn", "cnn_wide"):
    t = importlib.import_module(f"tasks.{name}")
    for attack in (False, True):
        w, norms = t.init(), []
        for r in range(t.ROUNDS):
            ref = t.train(w, *t.root_data(), 10_000 + r)
            raw = [t.train(w, *t.participant_data(i), 100 * r + i) for i in range(10)]
            norms += [float(np.linalg.norm(d)) for d in raw[N_MAL:]]
            ds = [clip(-10 * d if attack and i < N_MAL else d, 1.0) for i, d in enumerate(raw)]
            kept = keep([d @ ref for d in ds], [d @ d for d in ds])
            if len(kept) >= 2:
                w = w + np.mean([s * ds[i] for i, s in kept.items()], axis=0)
        q = np.percentile(norms, [10, 50, 90])
        print(f"{name:9} {len(w):7} params, {'sign-flip x10' if attack else 'no attack':13}: {t.evaluate(w)}; "
              f"honest |u| p10/50/90 {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f} (B = 1)", flush=True)
