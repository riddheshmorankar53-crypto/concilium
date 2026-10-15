"""Task for tabular data in CSV files: yes/no prediction (logistic regression), no coding needed.

Every party keeps its own copy of this file in its folder, next to its data, and all copies must say the same
in the settings below (agree on them first; the session id doesn't cover task files).
    participant:  my_data.csv   your rows (every party: same columns, a header row, numbers only)
    member:       root.csv      a small clean labeled set the committee trusts: the poisoning filter compares
                                every update with one trained on it. Keep it private to the committee.
                  test.csv      optional: the lead reports accuracy on it at the end (else on root.csv)
Run with TASK=csv_task.py (the file name in your folder): `TASK=csv_task.py sh concilium.sh participant ...`.

Settings: LABEL is the 0/1 column to predict; FEATURES the input columns (None: every other column). Features
should be on similar scales: run `python csv_task.py stats some.csv` on any representative file and paste the
MEAN and STD lines it prints below, so every party scales the same way.
"""
import csv
import os
import sys

import numpy as np

LABEL = "label"
FEATURES = None
MEAN = None
STD = None
ROUNDS = 20

HERE = os.path.dirname(os.path.abspath(__file__))  # data files sit next to this file


def read(name):
    """(X, y) from a CSV next to this file: the features, scaled, plus a constant 1 for the bias."""
    with open(os.path.join(HERE, name), newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{name}: no rows")
    features = FEATURES or [c for c in rows[0] if c != LABEL]
    try:
        X = np.array([[float(r[c]) for c in features] for r in rows])
        y = np.array([float(r[LABEL]) for r in rows])
    except (KeyError, ValueError) as e:
        raise ValueError(f"{name}: every row needs numbers in columns {features + [LABEL]} ({e})") from None
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError(f"{name}: column {LABEL} must hold only 0 and 1")
    if MEAN is not None:
        X = (X - np.array(MEAN)) / np.array(STD)
    return np.hstack([X, np.ones((len(X), 1))]), y


def n_features():
    with open(os.path.join(HERE, "my_data.csv" if os.path.exists(os.path.join(HERE, "my_data.csv")) else "root.csv"),
              newline="") as f:
        header = next(csv.reader(f))
    return len(FEATURES or [c for c in header if c != LABEL])


def init():
    return np.zeros(n_features() + 1)


def participant_data(i):
    # my_data_<i>.csv: the local demo runs every participant from one folder; a real participant has my_data.csv
    own = f"my_data_{i}.csv"
    return read(own if os.path.exists(os.path.join(HERE, own)) else "my_data.csv")


def root_data():
    return read("root.csv")


def train(w, X, y, seed):
    """5 epochs of full-batch gradient descent (deterministic, so seed isn't needed)."""
    w0 = w.copy()
    for _ in range(5):
        w = w - 0.1 * X.T @ (1 / (1 + np.exp(-X @ w)) - y) / len(y)
    return w - w0


def evaluate(w):
    name = "test.csv" if os.path.exists(os.path.join(HERE, "test.csv")) else "root.csv"
    X, y = read(name)
    return f"accuracy {float(((X @ w > 0) == y).mean()):.3f} on {name}"


if __name__ == "__main__" and sys.argv[1:2] == ["stats"]:
    MEAN = STD = None
    X = read(os.path.abspath(sys.argv[2]))[0][:, :-1]
    std = X.std(axis=0)
    print(f"MEAN = {[round(float(m), 6) for m in X.mean(axis=0)]}")
    print(f"STD = {[round(float(s), 6) if s > 0 else 1.0 for s in std]}")
