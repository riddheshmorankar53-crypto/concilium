"""Example task: logistic regression on sklearn's breast cancer data (31 parameters).

A task is everything deploy.py needs to know about your data and model. Copy this file, fill in the
functions, and run `TASK=path/to/your_task.py python deploy.py` (the file must be inside this folder,
since the containers only see this folder). The protocol never looks inside the model: it only sees
the flat float64 update vector that train() returns.
    ROUNDS                      how many training rounds
    init() -> w                 initial model as one flat float64 numpy vector
    participant_data(i) -> X, y participant i's own data (in a real deployment: read its local files)
    root_data() -> X, y         small clean labeled set held by the committee; the poisoning filter
                                compares every update with the update trained on it
    train(w, X, y, seed) -> u   one local training run from model w; returns new w - w. Must be
                                deterministic for a given seed (the simulation replays it)
    evaluate(w) -> str          optional: a one-line report the committee prints at the end
"""
import numpy as np

from fl_core import ROUNDS, accuracy, load, local_update

# ponytail: the demo splits one dataset between the participants; a real participant has only its own
_clients, _root, _test = load(non_iid=False)


def init():
    return np.zeros(_test[0].shape[1])


def participant_data(i):
    return _clients[i]


def root_data():
    return _root


def train(w, X, y, seed):
    return local_update(w, X, y)


def evaluate(w):
    return f"accuracy {accuracy(w, *_test):.3f}"
