"""Example task: eval_mnist.py's CNN on MNIST, non-IID (Dirichlet 0.5), 10 participants, seed 0 (14,410 parameters),
so the encrypted run can be set next to the plaintext table in RESEARCH.md. Needs mnist.npz in this folder:
    python eval_mnist.py data mnist     downloads it from OpenML once (the containers have no network for it)
The contract is in tasks/lr.py."""
import torch

import eval_mnist

torch.set_num_threads(1)  # deterministic CPU training, for the replay
_t = eval_mnist.task("mnist", True, 0)
ROUNDS = _t["rounds"]


def init():
    return _t["w0"].copy()


def participant_data(i):
    return _t["clients"][i]


def root_data():
    return _t["root"]


def train(w, X, y, seed):
    torch.manual_seed(seed)
    return _t["update"](w, X, y)


def evaluate(w):
    acc, worst = _t["evaluate"](w)
    return f"accuracy {acc:.3f}, worst-class recall {worst:.3f}"
