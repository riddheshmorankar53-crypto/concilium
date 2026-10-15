"""Example task: the chest X-ray CNN of tasks/cnn.py, wider (107,586 parameters, 14 ciphertexts per update; 7.6x).
For measuring how the protocol scales with model size. The contract is in tasks/lr.py."""
import torch
from torch import nn

import gap_b_cnn as c

ROUNDS = c.ROUNDS
torch.set_num_threads(1)  # deterministic CPU training, for the replay
torch.manual_seed(0)
_clients, _root, _test = c.load()
_net = nn.Sequential(nn.Conv2d(1, 16, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3), nn.ReLU(),
                     nn.MaxPool2d(2), nn.Flatten(), nn.Linear(800, 128), nn.ReLU(), nn.Linear(128, 2)).to(c.DEV)


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
