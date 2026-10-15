"""How often does the current filter collapse under gap E's attacks? python filter_reps.py <seed> <reps>"""
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
import gap_e

np.seterr(over="ignore", invalid="ignore")
seed, reps = int(sys.argv[1]), int(sys.argv[2])
t = gap_e.cnn_task(seed)
attacks = [("none", None), ("sign-flip x10", None), ("ALIE", None)] + \
    [("inside, hide", p) for p in ((45, 1.0, "hide"), (70, 2.9, "hide"), (85, 2.9, "hide"))] + \
    [("inside, flip", (70, 2.9, "flip"))]
done = set()
if len(sys.argv) > 3:  # resume: skip runs already in this log
    done = {tuple(l.split("|")[1:4]) for l in open(sys.argv[3]) if l.count("|") == 7}
for rep in range(reps):
    for kind, param in attacks:
        label = kind if param is None else f"{kind} {param[0]}deg {param[1]}x"
        for d in ("none", "filter"):
            if (str(rep), label, d) in done:
                continue
            torch.manual_seed(1000 * rep + 7)
            acc, rec, mal, hon, skip = gap_e.run(t, d, kind, param)
            print(f"{seed}|{rep}|{label}|{d}|{acc:.3f}|{rec:.3f}|{hon:.2f}|{skip}", flush=True)
