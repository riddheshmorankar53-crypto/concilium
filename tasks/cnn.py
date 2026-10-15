"""Example task: a small CNN on PneumoniaMNIST chest X-rays, non-IID (14,146 parameters).
Shows how to wrap a PyTorch model: flatten its parameters into the vector the protocol sees.
The contract is in tasks/lr.py."""
import torch

import gap_b_cnn as c

ROUNDS = c.ROUNDS
torch.set_num_threads(1)  # deterministic CPU training, for the replay
torch.manual_seed(0)
_clients, _root, _test = c.load()
_net = c.model()


def init():
    return c.parameters_to_vector(_net.parameters()).detach().double().cpu().numpy()


def participant_data(i):
    return _clients[i]


def root_data():
    return _root


def train(w, X, y, seed):
    torch.manual_seed(seed)
    return c.local_update(_net, w, X, y)


def evaluate(w):
    acc, rec = c.evaluate(_net, w, *_test)
    return f"accuracy {acc:.3f}, pneumonia recall {rec:.3f}"
