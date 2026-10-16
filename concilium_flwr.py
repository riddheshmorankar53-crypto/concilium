"""Concilium inside Flower: encrypted, poisoning-filtered aggregation for an existing Flower app.

Flower keeps running the rounds; Concilium's board and committee (deploy.py, unchanged) do the aggregation.
  client: app = ClientApp(mods=[concilium_mod]), node config concilium-config (the run's concilium.json) and
          concilium-key (this node's .key). The mod hands the user's train function the committee-signed model,
          takes the update from its reply, encrypts it to the committee and posts it (with its range proof) to
          the board, then strips the arrays from the reply: the Flower server never sees an update.
  server: ConciliumStrategy("concilium.json") in place of FedAvg. aggregate_train waits for the model t committee
          members signed and returns it. The Flower server is as untrusted as the board.
The committee still runs the task file (init, root_data, train, ROUNDS): it trains the filter's reference update on
the root data. Use the same train function in the ClientApp, and as many Flower rounds as the task's ROUNDS.
Checked by flower_check.py (FLOWER=1 python deploy.py).
"""
import json

import numpy as np
from flwr.app import ArrayRecord
from flwr.serverapp.strategy import FedAvg

import deploy

_JOINT = {}  # session -> (crypto context, joint public key): fetched and deserialized once per process


def flat(arrays):
    return np.concatenate([a.ravel().astype(np.float64) for a in arrays.to_numpy_ndarrays()])


def shaped(w, like):
    """w as an ArrayRecord with like's shapes and dtypes."""
    out, i = [], 0
    for a in like.to_numpy_ndarrays():
        out.append(w[i:i + a.size].reshape(a.shape).astype(a.dtype))
        i += a.size
    if i != len(w):
        raise ValueError(f"the model has {len(w)} parameters, the Flower arrays {i}")
    return ArrayRecord(out)


def model(rnd):
    """The model after round rnd: the task's start model, or the one t committee members signed."""
    return deploy.start_model(deploy.task()) if rnd == 0 else deploy.signed_model(rnd)


def concilium_mod(msg, ctx, call_next):
    kind = msg.metadata.message_type.split(".")[0]
    if kind not in ("train", "evaluate"):
        return call_next(msg, ctx)
    c = deploy.load(ctx.node_config["concilium-config"], ctx.node_config["concilium-key"], "participant")
    r = int(msg.content["config"]["server-round"])
    # Flower's round r trains Concilium's round r - 1 from model r - 1 and then evaluates model r
    w = model(r - 1 if kind == "train" else r)
    msg.content["arrays"] = shaped(w, msg.content["arrays"])  # the signed model, not whatever the server sent
    reply = call_next(msg, ctx)
    if kind == "evaluate" or reply.has_error():
        return reply
    if deploy.SESSION not in _JOINT:
        _JOINT[deploy.SESSION] = deploy.joint_key()
    from threshold import encrypt
    delta = deploy.prove(c, r - 1, flat(reply.content["arrays"]) - w)
    deploy.sput(f"post/{r - 1}/{c}", encrypt(*_JOINT[deploy.SESSION], delta))
    del reply.content["arrays"]  # the update went to the board encrypted; the Flower server gets metrics only
    return reply


class ConciliumStrategy(FedAvg):
    """FedAvg's sampling and evaluation; the aggregate is the committee's signed model. Every participant trains
    every round (the committee's --min decides how many must post)."""

    def __init__(self, config_path, **kw):
        with open(config_path) as f:
            deploy.CFG = json.load(f)
        deploy.check_config(deploy.CFG)
        deploy.SESSION = deploy.session(deploy.CFG)
        kw.setdefault("min_available_nodes", len(deploy.CFG["participants"]))
        super().__init__(fraction_train=1.0, **kw)
        self.like = None

    def configure_train(self, server_round, arrays, config, grid):
        self.like = arrays
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        list(replies)  # their arrays were stripped by the mod; the committee's model is the aggregate
        return shaped(deploy.signed_model(server_round), self.like), None
