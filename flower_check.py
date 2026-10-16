"""concilium_flwr.py checked: the demo's 10 participants as Flower nodes, against the real board and committee.

  FLOWER=1 python deploy.py        Flower's simulation runtime (Ray), one process per few nodes
  FLOWER=deploy python deploy.py   Flower's deployment runtime: a SuperLink, 10 SuperNodes, the app sent by `flwr run`
(Docker; builds the concilium-flower image once.) The launcher starts the board and the members as usual, and this
script instead of the 10 participant containers.
Each Flower node is a Concilium participant (partition i uses p<i>.key) whose ClientApp trains like
deploy.participant: same data, seeds and poisoners (0-2 send -10x), participant 0 lying in its range proof. So the
lead's end-of-run checks (every round matches plaintext defended FedAvg, the proofs reject exactly the liar, the run
survives member 2's crash) hold for updates that went through Flower. This script asserts the Flower side: no reply
reached the Flower server with an update in it, and its final model is the last one the committee signed.
"""
import json
import os
import shutil
import subprocess
import sys
import time

from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import ServerApp

import deploy
from concilium_flwr import ConciliumStrategy, concilium_mod, flat

RUN = os.environ.get("CONCILIUM_RUN", "/work/.concilium-demo")


def node_keys(msg, ctx, call_next):
    """Check only: partition i is participant i (a real node sets these in its node config), and participant 0 lies
    in its range proof, as in deploy.participant. Patched here, in the node's process: Ray ships this script's
    functions by value, so its module-level code never runs there."""
    if not hasattr(deploy.prove, "liar"):
        honest = deploy.prove
        deploy.prove = lambda c, rnd, delta, lie=False: honest(c, rnd, delta, lie=c == 0)
        deploy.prove.liar = True
    ctx.node_config["concilium-config"] = f"{RUN}/concilium.json"
    ctx.node_config["concilium-key"] = f"{RUN}/p{ctx.node_config['partition-id']}.key"
    return call_next(msg, ctx)


client = ClientApp(mods=[node_keys, concilium_mod])


@client.train()
def train(msg, ctx):
    t, c, rnd = deploy.task(), int(ctx.node_config["partition-id"]), int(msg.content["config"]["server-round"]) - 1
    w = flat(msg.content["arrays"])
    X, y = t.participant_data(c)
    delta = t.train(w, X, y, deploy.seed(rnd, c))
    if c < deploy.POISON:
        delta = deploy.POISON_SCALE * delta
    return Message(RecordDict({"arrays": ArrayRecord([w + delta]), "metrics": MetricRecord({"num-examples": len(y)})}),
                   reply_to=msg)


class Checked(ConciliumStrategy):
    def aggregate_train(self, server_round, replies):
        replies = list(replies)
        assert replies and all("arrays" not in r.content for r in replies if not r.has_error()), \
            "an update reached the Flower server"
        return super().aggregate_train(server_round, replies)


server = ServerApp()


@server.main()
def main(grid, ctx):
    strategy = Checked(f"{RUN}/concilium.json", fraction_evaluate=0.0)
    t = deploy.task()
    try:
        result = strategy.start(grid=grid, initial_arrays=ArrayRecord([deploy.start_model(t)]), num_rounds=t.ROUNDS)
        assert (flat(result.arrays) == deploy.signed_model(t.ROUNDS)).all(), "Flower's final model must be the signed one"
    except BaseException as e:
        with open(f"{RUN}/flower.fail", "w") as f:  # neither runtime hands a failed ServerApp's error to its caller
            f.write(f"{type(e).__name__}: {e}")
        raise
    print(f"flower: {t.ROUNDS} rounds, no update reached the Flower server, final model = the committee's signed model",
          flush=True)
    open(f"{RUN}/flower.ok", "w").close()


PYPROJECT = """[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "concilium-check"
version = "1.0.0"
dependencies = []

[tool.flwr.app]
publisher = "concilium"

[tool.flwr.app.components]
serverapp = "flower_check:server"
clientapp = "flower_check:client"
"""


def deployment(n):
    """One SuperLink, n SuperNodes (node i is participant i), this file sent to them as a Flower app by `flwr run`."""
    app, home = f"{RUN}/app", f"{RUN}/flwr-home"
    os.makedirs(app)
    os.makedirs(home)
    shutil.copy(__file__, app)
    with open(f"{app}/pyproject.toml", "w") as f:
        f.write(PYPROJECT)
    with open(f"{home}/config.toml", "w") as f:
        f.write('[superlink]\ndefault = "check"\n\n[superlink.check]\naddress = "127.0.0.1:8000"\ninsecure = true\n')
    env = os.environ | {"FLWR_HOME": home}
    procs = [subprocess.Popen(["flower-superlink", "--insecure"], env=env)]
    procs += [subprocess.Popen(["flower-supernode", "--insecure", "--port", str(9200 + i),
                                "--node-config", f"partition-id={i}"], env=env) for i in range(n)]
    try:
        start = time.time()  # the SuperLink's HTTP API up (the SuperNodes retry on their own)
        while subprocess.run(["python3", "-c", "import socket; socket.create_connection(('127.0.0.1', 8000), 1)"],
                             capture_output=True).returncode and time.time() - start < 120:
            time.sleep(1)
        subprocess.run(["flwr", "run", app, "check", "--stream"], env=env, check=True, timeout=3600)
        start = time.time()  # --stream may return before the ServerApp's last line
        while not any(os.path.exists(f"{RUN}/flower.{s}") for s in ("ok", "fail")) and time.time() - start < 600:
            time.sleep(1)
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    with open(f"{RUN}/concilium.json") as f:
        n = len(json.load(f)["participants"])
    if sys.argv[1:] == ["deploy"]:
        deployment(n)
    else:
        from flwr.simulation import run_simulation
        # 4 nodes at a time: each holds the crypto context, and the members need the rest of the CPU and RAM
        run_simulation(server, client, num_supernodes=n,
                       backend_config={"init_args": {"num_cpus": 4}, "client_resources": {"num_cpus": 1}})
    if os.path.exists(f"{RUN}/flower.fail"):
        print(open(f"{RUN}/flower.fail").read(), file=sys.stderr)
    sys.exit(0 if os.path.exists(f"{RUN}/flower.ok") else 1)
