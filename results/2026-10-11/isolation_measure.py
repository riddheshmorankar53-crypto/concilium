"""How often does a decrypted round hold exactly one honest update next to colluders' (decision D, 2026-10-11)?
3 colluding participants (0..2) send honest-looking updates (trained on their own data, so the filter keeps them
like anyone else) and know them; a malicious board shrinks the roster to the colluders + h honest participants.
The round's sum is decrypted when >= 2 updates are kept (today's floor). If exactly 1 honest update is kept,
the colluders know every other term of the published change (up to the clip factors).
    ~/ppfl-gpu/Scripts/python results/2026-10-11/isolation_measure.py <seed> <draws>
Output: seed|draw|h|decrypted rounds|rounds with 1 honest kept|rounds with 0 honest kept|acc"""
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
import gap_e  # noqa: E402
from fl_core import N_CLIENTS, N_MAL, keep  # noqa: E402

np.seterr(over="ignore", invalid="ignore")
seed, draws = int(sys.argv[1]), int(sys.argv[2])
t = gap_e.cnn_task(seed)
for rep in range(draws):
    for h in (3, 4, 5):  # roster minimum 6, 7, 8
        torch.manual_seed(1000 * rep + 7)
        rng = np.random.default_rng(1000 * rep + seed)
        w, dec, one, zero = t["w0"].copy(), 0, 0, 0
        for _ in range(t["rounds"]):
            ref = t["update"](w, *t["root"])
            deltas = [t["update"](w, X, y) for X, y in t["clients"]]
            roster = list(range(N_MAL)) + sorted(rng.choice(range(N_MAL, N_CLIENTS), h, replace=False).tolist())
            ds = [deltas[c] for c in roster]
            kept = {roster[i]: s for i, s in keep([d @ ref for d in ds], [d @ d for d in ds]).items()}
            if len(kept) < 2:
                continue
            dec += 1
            honest = sum(c >= N_MAL for c in kept)
            one, zero = one + (honest == 1), zero + (honest == 0)
            w = w + np.mean([s * deltas[c] for c, s in kept.items()], axis=0)
        print(f"{seed}|{rep}|{h}|{dec}|{one}|{zero}|{t['evaluate'](w)[0]:.3f}", flush=True)
