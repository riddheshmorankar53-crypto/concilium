"""threshold.py across machines: 1 untrusted board (the server), a committee of members, participants.

Everything goes through the board over HTTP(S), so it can count every byte. The board is untrusted: every
message on it is signed by its author (Ed25519) and checked against the public keys in the shared config
file, bound to this deployment and to its board slot. So the board can drop or delay messages but not forge
or swap them (a swapped joint public key would let it decrypt every update; a swapped noise ciphertext or
aggregate partial would add an update the filter never saw; a swapped model would retarget the participants).
Committee-only messages (roster, score partials, fallback votes) are also sealed with AES-GCM under a
committee key that the lead hands each member through X25519: anyone holding all score partials can fuse them.
Each member generates its own key share (chained, as in threshold.setup) and posts only public keys.
Rounds reuse threshold.Member. Dropouts: the lead closes each round `deadline` s after
min_participants have posted, and posts the roster; all members use it, so they score the same ciphertexts.
Member dropouts: with threshold t < n, each member Shamir-shares its key share to the others after keygen; a
member that posts no noise within member_timeout is dropped for good, and the rest fold their shares of its key
into their own (threshold.Member.recover). The run goes on while >= t members are left. Members also drop
mid-round, the lead included: every member posts a heartbeat, and every wait on a member's message has one
decider (the lowest member other than its author) who rules "ok" when it arrives or "gone" once the author's
heartbeat stops for member_timeout; the next lowest rules on that ruling, and everyone follows it (deploy.member's
need() has the one corner where members could disagree); on "gone" the
round is abandoned (model unchanged), the member dropped, and the lowest member left leads. A dropped member
rejoins by restarting with the same command: its key share and the committee key are in <key file>.state
(owner-only, written at keygen), the rest comes from the board; it asks in a signed rejoin/{rnd}/{i} and the lead
admits it in that round's member list. Only members ranked after the current lead rejoin (a returning former lead
would take the lead back without the run's history).
The data and model come from a task file (contract in tasks/lr.py); the protocol only sees update vectors.
Range proofs (config --norm-bound B, range_proof.py): each participant also posts PINE shares of its update,
sealed per member, proving |u| <= B; the members check the ciphertext holds the same u (random projections from
committed-then-revealed member randomness) and score only those that pass, with flooding sized for B. PINE shares
are n-of-n, so a round in which the member set changes is skipped, as is one with < min_participants proven.

Real deployment, each party on its own machine (with the concilium image, or Linux with openfhe installed):
    python deploy.py keys alice                  # every party: alice.key (secret, never leaves) + alice.pub
    python deploy.py config concilium.json --board https://board.example:8000 --task my_task.py \\
        --committee m0.pub m1.pub m2.pub --participants p0.pub p1.pub ...   # organizer, then send concilium.json to all
    python deploy.py board concilium.json [cert.pem key.pem]  # the server; HTTPS if given (others: SSL_CERT_FILE for a private CA)
    python deploy.py member concilium.json m1.key     # each committee member (the first one listed leads)
    python deploy.py participant concilium.json p0.key   # TASK=... overrides the config's task file (local data paths)
Everyone must hold the identical concilium.json: it fixes who is who. Compare the session id it prints.
A run that can't go on (too few members left to decrypt) restarts as a new run from its last signed model:
    python deploy.py export concilium.json model.bin  # anyone, while the old board is up: the latest model t members signed
    python deploy.py config new.json ... --init-model model.bin  # the model goes into the config (and the session id)
then new keys are made as usual: a new board, keygen and rounds, starting from that model.
Local demo, one host, 1 + 3 + 10 containers (generates all keys and the config itself):
    docker build --target tool -t concilium .   # or: docker pull ghcr.io/riddheshmorankar53-crypto/concilium + docker tag
    python deploy.py                       # tasks/lr.py, prints the lead's report
    TASK=cnn python deploy.py              # tasks/cnn.py: chest X-ray CNN (14k parameters, 2 ciphertexts per update)
    TASK=my_task.py python deploy.py       # your own task file (inside this folder)
    SIMULATE=0 python deploy.py            # no simulated poisoning, dropouts or plaintext replay
    NORM_BOUND=0 POISON_SCALE=1e12 python deploy.py  # huge updates: exercises the fallback (gap_f.py; with
                                           # range proofs on, the proofs reject them before scoring)
    TLS=1 python deploy.py                 # board over HTTPS, throwaway self-signed certificate
    TAMPER=noise/0/1 python deploy.py      # the board alters that message: the run must abort, not go on
    CRASH=lead python deploy.py            # the lead crashes at the start of the middle round (default: last member)
    CRASH=lead-mid python deploy.py        # ... or mid-round, after its score partials (also member-mid)
    CRASH=successor-mid python deploy.py   # member 1, who rules on the lead's messages, crashes mid-round
    MEMBERS=4 THRESHOLD=2 CRASH=lead+successor-mid python deploy.py  # the lead and member 1 crash together
    CRASH=lead+successor python deploy.py  # same with 3 members: too few to decrypt, member 2 aborts the run (exit 1)
    EXPORT=model.bin CRASH=lead+successor python deploy.py; INIT_MODEL=model.bin python deploy.py  # ... and the run
                                           # resumes as a new one from its last signed model (RESUME=1 does both)
    CRASH=member-rejoin python deploy.py   # the last member crashes, then restarts and rejoins
    REPLAY_BEAT=1 CRASH=member-mid python deploy.py  # the board serves random old heartbeats: the crashed member
                                           # must still be dropped
    BAD_POST=1 python deploy.py            # participant 3 posts junk (round 2), a copy of participant 4's post (round 4)
                                           # and bytes that crash OpenFHE (round 6, post_crash_check.py)
    WITHHOLD=1 python deploy.py            # the 3 poisoners stop posting at round 5: rounds go on with 6 (participant 8
                                           # leaves at 12) until participant 7 leaves at 15: 3 skipped rounds, the run stops
    DP_Z=0 python deploy.py                # no masking noise on the sums (default 0.1 x the norm bound)
    NORM_BOUND=0 python deploy.py          # range proofs off (on by default, B = 1: ~6 s rounds on LR; participant 0 lies)
"""
import base64
import hashlib
import json
import os
import shutil
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fl_core import N_CLIENTS, N_MAL

PORT, NET, IMAGE = int(os.environ.get("PORT", 8000)), "concilium-net", os.environ.get("IMAGE", "concilium")  # demo: or concilium
# the demo's simulation: N_MAL participants poison (sign-flip x10), two drop out, and the lead replays every
# round in plaintext to check it (possible only because the demo holds everyone's data)
SIMULATE = os.environ.get("SIMULATE", "1" if len(sys.argv) < 2 else "0") == "1"  # on for the demo launcher only
POISON = N_MAL if SIMULATE else 0
POISON_SCALE = float(os.environ.get("POISON_SCALE", -10))  # the poisoners send this x their update (1e12: gap_f's DoS)
OFFLINE = {9: range(5, 10), 8: range(12, 100)} if SIMULATE else {}  # participant -> rounds it misses
CFG, ME, SESSION, KEY_PATH = None, None, b"", None  # the config, my private keys, the deployment id, my key file
PARTY = ""  # who I am on the board: m<i> or p<c>
SIG = 64  # Ed25519 signature bytes
TAMPER = os.environ.get("TAMPER")  # demo only: the board flips a byte of the message it stores under this key
NORM_BOUND = os.environ.get("NORM_BOUND", "1")  # demo launcher: range proofs with this bound; 0 turns them off
WITHHOLD = os.environ.get("WITHHOLD") == "1"  # demo: poisoners withhold from round 5, participant 7 leaves at 15
BAD_POST = os.environ.get("BAD_POST") == "1"  # demo: participant 3 sends invalid posts (gap_h.py)
REPLAY_BEAT = os.environ.get("REPLAY_BEAT") == "1"  # demo: the board answers heartbeat reads with random old ones
CRASH = os.environ.get("CRASH", "member")  # demo: member, lead, member-mid or lead-mid crashes halfway


def crashed_members(n):
    """Who CRASH crashes: the last member by default, or the lead, its successor, or both (lead+successor)."""
    return {"lead": [0], "successor": [1], "lead+successor": [0, 1]}.get(CRASH.split("-")[0], [n - 1])


# --- the board: an untrusted key-value store that counts traffic per message kind ---

PARTICIPANT_PUT_MAX = 64 << 20  # bytes: a post is a few MB; the board checks the size before reading the body


def may_write(party, key, data, sig, latest):
    """Board: a write must be signed by a party in the config. A participant may write only its own post and proof
    slots of the current round (at most one past the latest roster) and its quit slot, each once (an identical
    re-send is fine), so it can't show members different posts or fill the board. Nobody else writes those slots.
    Before 2026-10-11 the board stored any write, so one participant could stop the run (gap_h.py). The board stays
    untrusted: a malicious one can still stop it, or show members different bytes."""
    try:
        member, idx = party[0] == "m", int(party[1:])
        author = CFG["committee"][idx]["sign"] if member else CFG["participants"][idx] if party[0] == "p" else None
        if author is None or idx < 0 or not verify(author, f"PUT {key}", data, bytes.fromhex(sig)):
            return False
        k = key.split("/")
        own = ((len(k) == 3 and k[0] in ("post", "proof") and k[2] == str(idx)
                and max(0, latest - 1) <= int(k[1]) <= latest + 1)  # not older: pruned slots must stay empty
               or key == f"quit/{SESSION.hex()}/{idx}")
    except (ValueError, IndexError):
        return False
    return (member and not key.startswith(("post/", "proof/", "quit/"))) or (not member and own)


BULKY = ("post", "proof", "noise", "pine", "pineout", "score", "agg", "fallback", "each")  # per-round, pruned


class Board(BaseHTTPRequestHandler):
    store, up, down, written, latest, beats = {}, Counter(), Counter(), {}, [-1], {}  # written: participant slot -> hash
    timeout = 60  # s per socket read/write: a stalled client only ties up its own thread

    def do_PUT(self):
        size, party = int(self.headers["Content-Length"]), self.headers.get("X-Party", "")
        if size > (PARTICIPANT_PUT_MAX if not party.startswith("m") else 1 << 30):  # don't even read it
            self.send_response(413)
            self.end_headers()
            self.close_connection = True
            return
        data = self.rfile.read(size)
        if len(data) != int(self.headers["Content-Length"]):  # the uploader gave up mid-transfer: it retries
            return
        key = self.path[1:]
        digest = hashlib.sha256(data).digest()
        if (not may_write(party, key, data, self.headers.get("X-Sig", ""), self.latest[0])
                or self.written.setdefault(key, digest) != digest):  # participant slots: setdefault only for them
            print(f"board: refused {party or '?'} PUT {key} (latest roster {self.latest[0]}, "
                  f"{'already written' if key in self.written else 'not allowed'})", file=sys.stderr, flush=True)
            self.send_response(403)
            self.end_headers()
            return
        if not party.startswith("p"):
            self.written.pop(key, None)  # write-once applies to participants' slots only
        if key.startswith(("roster/", "model/")):  # model/r opens round r (r - 1 had no roster if its lead crashed)
            opened = int(key.split("/")[1]) - key.startswith("model/")
            if opened > self.latest[0]:
                self.latest[0] = opened
                self.prune(opened - 2)
        if key == TAMPER:  # a malicious board: every reader must notice (bad signature or AEAD tag)
            data = data[:len(data) // 2] + bytes([data[len(data) // 2] ^ 1]) + data[len(data) // 2 + 1:]
        self.store[key] = data
        if REPLAY_BEAT and key.startswith("beat/"):
            self.beats.setdefault(key, []).append(data)
        self.up[key.split("/")[0]] += len(data)
        self.send_response(204)
        self.end_headers()

    def prune(self, last):
        """Drop the bulky per-round messages of rounds 1..last: nobody reads a round 2 behind the current one, and
        kept forever they grew past the host's memory at 107k parameters (14 ciphertexts per update). Signatures,
        rosters, models and member lists stay (small; a rejoining member reads them). Round 0 stays for the
        demo's check that the board can't read the score partials."""
        for key in list(self.store):  # other threads write meanwhile: iterate over a copy
            kind, _, rest = key.partition("/")
            if kind in BULKY and 0 < int(rest.split("/")[0]) <= last:
                self.store.pop(key, None)

    def do_GET(self):
        key = self.path[1:]
        body = json.dumps({"up": self.up, "down": self.down}).encode() if key == "stats" else self.store.get(key)
        if key in self.beats and os.urandom(1)[0] < 128:  # REPLAY_BEAT: half the time an old heartbeat, as a
            body = self.beats[key][int.from_bytes(os.urandom(4), "big") % len(self.beats[key])]  # replayer would
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        if key != "stats":
            self.down[key.split("/")[0]] += len(body)
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):  # "is it posted yet?" without downloading it
        self.send_response(200 if self.path[1:] in self.store else 404)
        self.end_headers()

    def log_message(self, *args):
        pass


# every request gets a timeout: a NAT or port forwarder that drops a connection attempt otherwise stalls the caller
# for a full TCP backoff (19-36 s per drop seen through Docker Desktop's forwarder); timing out and retrying is faster.
# A stalled transfer restarts its whole file, so a remote member on a slow link sets it longer (friend/3_run: 60)
NET_TIMEOUT = float(os.environ.get("NET_TIMEOUT", 5))


def put(key, data, timeout=1800):
    start = time.time()
    while True:
        try:
            auth = {"X-Party": PARTY, "X-Sig": ME[0].sign(signed(f"PUT {key}", data)).hex()}  # the board checks it
            return urllib.request.urlopen(urllib.request.Request(f"{CFG['board']}/{key}", data=data, method="PUT",
                                                                 headers=auth), timeout=NET_TIMEOUT).read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:  # HTTPError (a refusal) is a URLError
            # too: don't retry it. ConnectionError: reset or closed without a reply (raised raw, not as URLError);
            # resending is safe, the board accepts an identical re-send
            if isinstance(e, urllib.error.HTTPError) or isinstance(getattr(e, "reason", None), ssl.SSLError) \
                    or time.time() - start > timeout:
                raise


def try_get(key, method="GET"):
    try:
        return urllib.request.urlopen(urllib.request.Request(f"{CFG['board']}/{key}", method=method),
                                      timeout=NET_TIMEOUT).read()
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    except urllib.error.URLError as e:
        if isinstance(e.reason, ssl.SSLError):
            raise  # wrong certificate: waiting won't fix it
        # board not up yet, or the connection attempt timed out
    except (TimeoutError, ConnectionError):  # stalled mid-transfer, or reset / closed without a reply: ask again
        pass
    return None


def get(key, timeout=1800):
    start, polls = time.time(), 0
    while (data := try_get(key)) is None:
        if time.time() - start > timeout:
            raise TimeoutError(f"nothing posted at {key}")
        polls += 1
        if polls % 20 == 0:
            check_abort()
        time.sleep(0.05)
    return data


class Gone(Exception):
    """A member's decider ruled it gone (deploy.member's need()); args[0] = its index."""


def arrived(name, j, timeout=None):
    """Decider: True once name is posted; False if member j's heartbeat stops advancing for timeout (default
    member_timeout) first.
    Heartbeats are signed counts (ns timestamps, so they go on rising after a restart) and only a higher one counts:
    replayed old heartbeats (plain HTTP, or the board) can't keep a dead member alive (REPLAY_BEAT=1). Before
    2026-10-11 any change counted, so alternating two old ones did."""
    last, seen = -1, time.time()
    while try_get(name, "HEAD") is None:
        blob = try_get(f"beat/{j}")
        try:
            beat = int(blob[:-SIG]) if verify(CFG["committee"][j]["sign"], f"beat/{j}", blob[:-SIG], blob[-SIG:]) else -1
        except (TypeError, ValueError):  # not posted yet, or not a count
            beat = -1
        if beat > last:
            last, seen = beat, time.time()
        elif time.time() - seen > (timeout or CFG["member_timeout"]):
            return False
        check_abort()
        time.sleep(0.05)
    return True


def check_abort():
    """Stop if a member gave up. Only a member's signed abort counts (gap_h.py): a participant that fails just quits."""
    name = f"abort/{SESSION.hex()}"  # per session: a leftover process can't stop a new run
    if (blob := try_get(name)) is not None and any(verify(m["sign"], name, blob[:-SIG], blob[-SIG:]) for m in CFG["committee"]):
        raise SystemExit(f"aborted by {blob[:-SIG].decode(errors='replace')}")


def run_role(fn, role, args):
    """Run a member or participant; if it fails, tell the others instead of leaving them waiting. A member stops the
    run; a participant only quits, and the lead stops the run once too few participants are left to fill a round."""
    i = load(*args, role)
    try:
        fn(i)
    except Exception as e:
        why = f"{role} {i}: {type(e).__name__}: {e}".encode()
        try:
            sput(f"abort/{SESSION.hex()}" if role == "member" else f"quit/{SESSION.hex()}/{i}", why)
        except Exception:
            pass  # another member's abort is already there, or the board is gone
        raise


class BoardServer(ThreadingHTTPServer):
    # pending connections the OS holds for accept(); the default 5 overflowed with 9 parties polling through
    # Docker's port proxy on Linux, which then reset connections (CI, 2026-10-16)
    request_queue_size = 128


def board_server(cert=None, key=None, host=""):
    server = BoardServer((host, PORT), Board)
    if cert:
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(cert, key)
        # the handshake runs in each request's thread, under Board.timeout; on accept() it ran in the one
        # serving thread, so a client that never finished it stalled the board for everyone (board_tls_check.py)
        server.socket = tls.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)
    return server


# --- identities, config, signed and sealed messages ---

def keys(path):
    """Write path.key (my private keys, owner-only) and path.pub (what goes into the config)."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    s, d = Ed25519PrivateKey.generate(), X25519PrivateKey.generate()
    with os.fdopen(os.open(f"{path}.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
        json.dump({"sign": s.private_bytes_raw().hex(), "dh": d.private_bytes_raw().hex()}, f)
    with open(f"{path}.pub", "w") as f:
        json.dump({"sign": s.public_key().public_bytes_raw().hex(), "dh": d.public_key().public_bytes_raw().hex()}, f)


# What every party runs (the Dockerfile copies the same files, and pine_poc/): its hash is part of the session id
CODE = ("deploy.py", "threshold.py", "fl_core.py", "range_proof.py", "gap_c.py", "gap_b.py", "gap_b_cnn.py")


@lru_cache(maxsize=None)
def code_hash():
    """sha256 over the tool's source (CODE and pine_poc's .py files; CRLF read as LF, so a Windows checkout
    matches). Task files aren't in it: each party may adapt its own (data paths)."""
    root = os.path.dirname(os.path.abspath(__file__))
    pine = sorted(os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/")
                  for d, _, names in os.walk(os.path.join(root, "pine_poc")) for f in names if f.endswith(".py"))
    h = hashlib.sha256()
    for f in list(CODE) + pine:
        with open(os.path.join(root, f), "rb") as src:
            h.update(f.encode() + b"\0" + src.read().replace(b"\r\n", b"\n") + b"\0")
    return h.digest()


def session(cfg):
    """The config and the code: parties running different code (a stale or modified build) get different session
    ids, which they compare at start, and can't verify each other's signatures, so the run stops at once."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode() + code_hash()).digest()


def check_config(cfg):
    n = len(cfg["participants"])
    signers = [m["sign"] for m in cfg["committee"]] + cfg["participants"]
    if len(cfg["committee"]) < 2:
        raise ValueError("a committee needs >= 2 members (1 would be a single key holder, gap C)")
    if len(set(signers)) != len(signers):
        raise ValueError("every party needs its own key")
    if not 3 <= cfg["min_participants"] <= n:  # a server + (min - 2) colluders can isolate one update
        raise ValueError(f"min_participants must be in 3..{n}")
    if cfg["min_participants"] > n - cfg.get("max_malicious", 0):  # older configs have no max_malicious
        raise ValueError(f"min_participants must be <= n - max_malicious = {n - cfg['max_malicious']}: "
                         "otherwise attackers that withhold their posts stall every round")
    if not 2 <= threshold(cfg) <= len(cfg["committee"]):
        raise ValueError(f"threshold must be in 2..{len(cfg['committee'])} (1 would let any member decrypt alone)")
    if cfg.get("norm_bound") is not None and not cfg["norm_bound"] > 0:
        raise ValueError("norm_bound must be > 0")
    if cfg.get("dp_z", 0) and cfg.get("norm_bound") is None:  # older configs have no dp_z
        raise ValueError("dp_z needs norm_bound: without it nothing bounds an update, so no noise size hides one")
    if not cfg.get("dp_z", 0) >= 0:
        raise ValueError("dp_z must be >= 0")


def threshold(cfg):
    """t: any t members can decrypt (so t colluders can too), and the run survives n - t member dropouts."""
    return cfg.get("threshold", len(cfg["committee"]))


def config(argv):
    import argparse
    a = argparse.ArgumentParser(prog="deploy.py config", description="write the shared deployment config")
    a.add_argument("out")
    a.add_argument("--board", required=True, help="URL everyone reaches the board at, e.g. https://host:8000")
    a.add_argument("--task", required=True, help="task name (tasks/<name>.py) or .py path")
    a.add_argument("--committee", nargs="+", required=True, help="members' .pub files; the first leads")
    a.add_argument("--participants", nargs="+", required=True, help="participants' .pub files")
    a.add_argument("--max-malicious", type=int, help="participants that may be attackers (default 30%%)")
    a.add_argument("--min", type=int, help="participants needed to run a round (default n - max-malicious - 1, at "
                   "least 3): the honest ones alone, less one dropout, must be able to fill a round")
    a.add_argument("--round-timeout", type=float, default=300.0, help="s to wait for --min posts; then the round is "
                   "skipped, and after 3 skipped in a row the run stops")
    a.add_argument("--deadline", type=float, default=3.0, help="s to wait for stragglers once min have posted")
    a.add_argument("--threshold", type=int, help="members needed to decrypt (default: all but one, at least 2). "
                   "Lower survives more dropouts, but that many colluding members can decrypt every update")
    a.add_argument("--member-timeout", type=float, default=60.0, help="s before a silent member counts as dropped")
    a.add_argument("--norm-bound", type=float, default=1.0, help="participants prove |update| <= this (PINE, "
                   "range_proof.py); honest ones clip to it. Pick it from your task's honest update norms (default 1)")
    a.add_argument("--dp-z", type=float, default=0.1, help="each member adds Gaussian noise of std dp_z x norm-bound "
                   "to each round's sum, so colluders can't read back a lone honest update even with all but one "
                   "member (3 members: 0.17 in all, ~3 points on the CNN's smallest rosters; it's a mask, not a DP "
                   "guarantee: eps ~ 48 per honest member's noise per round at delta 1e-5; z = 1 gives 4.8 but costs "
                   "15-30 points). Needs the norm bound; 0 turns it off")
    a.add_argument("--no-norm-bound", action="store_true", help="no range proofs: faster, but a participant can then "
                   "repost a transform of another's ciphertext and the members read parts of it (gap_i.py)")
    a.add_argument("--init-model", help="start from this model (from `export`: a run that stopped) instead of the "
                   "task's init; it's stored in the config, so the session id covers it")
    a = a.parse_args(argv)
    pub = lambda p: json.load(open(p))
    n = len(a.participants)
    cfg = {"board": a.board.rstrip("/"), "task": a.task, "committee": [pub(p) for p in a.committee],
           "participants": [pub(p)["sign"] for p in a.participants],
           "max_malicious": n * 3 // 10 if a.max_malicious is None else a.max_malicious,
           "min_participants": a.min or max(3, n - (n * 3 // 10 if a.max_malicious is None else a.max_malicious) - 1),
           "deadline": a.deadline, "round_timeout": a.round_timeout,
           "threshold": a.threshold or max(2, len(a.committee) - 1), "member_timeout": a.member_timeout,
           "norm_bound": None if a.no_norm_bound else a.norm_bound, "dp_z": 0.0 if a.no_norm_bound else a.dp_z}
    if a.init_model:
        cfg["init_model"] = base64.b64encode(open(a.init_model, "rb").read()).decode()
    check_config(cfg)
    with open(a.out, "w") as f:
        json.dump(cfg, f, indent=1)
    print(f"{a.out}: {len(cfg['committee'])} members ({cfg['threshold']} decrypt), {n} participants, "
          f"session {session(cfg).hex()[:16]}")


def load(cfg_path, key_path, role):
    """Read the config and my keys; returns my index among the committee or the participants."""
    global CFG, ME, SESSION, KEY_PATH, PARTY
    KEY_PATH = key_path
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    with open(cfg_path) as f:
        CFG = json.load(f)
    check_config(CFG)
    with open(key_path) as f:
        k = json.load(f)
    ME = (Ed25519PrivateKey.from_private_bytes(bytes.fromhex(k["sign"])), X25519PrivateKey.from_private_bytes(bytes.fromhex(k["dh"])))
    SESSION = session(CFG)
    mine = ME[0].public_key().public_bytes_raw().hex()
    ids = [m["sign"] for m in CFG["committee"]] if role == "member" else CFG["participants"]
    if mine not in ids:
        raise SystemExit(f"{key_path} is not a {role} in {cfg_path}")
    PARTY = f"{role[0]}{ids.index(mine)}"
    print(f"{role} {ids.index(mine)}, session {SESSION.hex()[:16]}", file=sys.stderr)
    return ids.index(mine)


def signed(name, data):
    """What an author signs: bound to this deployment and this board slot, so it can't be replayed elsewhere.
    The data is prehashed: keygen messages are hundreds of MB, and Ed25519 would hash them 2-3 times."""
    return SESSION + hashlib.sha256(name.encode()).digest() + hashlib.sha256(data).digest()


@lru_cache(maxsize=None)
def public(hex_key):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    return Ed25519PublicKey.from_public_bytes(bytes.fromhex(hex_key))


def verify(author, name, data, sig):
    from cryptography.exceptions import InvalidSignature
    try:
        public(author).verify(sig, signed(name, data))
        return True
    except InvalidSignature:
        return False


def sput(name, data):
    put(name, data + ME[0].sign(signed(name, data)))


def sget(name, author):
    """Fetch name; it must carry author's signature (author: hex public key)."""
    blob = get(name)
    if not verify(author, name, blob[:-SIG], blob[-SIG:]):
        raise PermissionError(f"bad signature on {name}")
    return blob[:-SIG]


def mget(name, j):
    return sget(name, CFG["committee"][j]["sign"])


def from_members(name):
    """name, signed by any committee member (for what the lead posts, when a reader doesn't know who leads)."""
    blob = get(name)
    if not any(verify(m["sign"], name, blob[:-SIG], blob[-SIG:]) for m in CFG["committee"]):
        raise PermissionError(f"{name} isn't signed by a committee member")
    return blob[:-SIG]


def seal(key, name, data):
    """AES-GCM; the board key name is bound in, so a blob can't be moved to another slot."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    nonce = os.urandom(12)
    return nonce + AESGCM(key).encrypt(nonce, data, name.encode())


def unseal(key, name, blob):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM(key).decrypt(blob[:12], blob[12:], name.encode())


def kdf(shared, info):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    return HKDF(hashes.SHA256(), 32, SESSION, info).derive(shared)


def x25519(raw):
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey
    return X25519PublicKey.from_public_bytes(raw)


@lru_cache(maxsize=None)
def pairwise(j):
    """An AES key only member j and I can derive (X25519)."""
    return kdf(ME[1].exchange(x25519(bytes.fromhex(CFG["committee"][j]["dh"]))), b"concilium committee key")


def to_member(j, name, data):
    """Participant -> member j, sealed under a one-time X25519 key (the participant signs the whole post).
    Participants are in the config by signing key only, so there is no static key pair to derive from."""
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    eph = X25519PrivateKey.generate()
    key = kdf(eph.exchange(x25519(bytes.fromhex(CFG["committee"][j]["dh"]))), b"concilium participant share")
    return eph.public_key().public_bytes_raw() + seal(key, name, data)


def from_participant(name, blob):
    return unseal(kdf(ME[1].exchange(x25519(blob[:32])), b"concilium participant share"), name, blob[32:])


def committee_key(i):
    """The lead picks it and seals it to each member under an X25519 key only the two of them can derive."""
    if i:
        return unseal(pairwise(0), f"committee/{i}", get(f"committee/{i}"))
    key = os.urandom(32)
    for j in range(1, len(CFG["committee"])):
        put(f"committee/{j}", seal(pairwise(j), f"committee/{j}", key))
    return key


# --- rounds ---

def close_round(rnd):
    """Lead: who's in this round. Everyone after the deadline if enough posted; [] (the round is skipped) if fewer
    than min_participants posted within round_timeout, e.g. attackers withholding their posts."""
    n, least = len(CFG["participants"]), CFG["min_participants"]
    start = time.time()
    enough, posted, polled = None, set(), 0.0  # the deadline runs from when `least` have posted, not from the start
    while len(posted) < n and (enough is None or time.time() - enough < CFG["deadline"]):
        posted |= {c for c in range(n) if c not in posted and try_get(f"post/{rnd}/{c}", "HEAD") is not None
                   and (CFG.get("norm_bound") is None or try_get(f"proof/{rnd}/{c}", "HEAD") is not None)}
        if enough is None and len(posted) >= least:
            enough = time.time()
        if enough is None and time.time() - polled > 1:  # participants that quit for good (run_role)
            polled = time.time()
            gone = [c for c in range(n) if c not in posted and try_get(f"quit/{SESSION.hex()}/{c}", "HEAD") is not None]
            if len(gone) > n - least:
                raise PermissionError(f"participants {gone} quit: fewer than {least} are left to post")
        if enough is None and time.time() - start > CFG.get("round_timeout", 300):
            return []
        check_abort()
        time.sleep(0.05)
    return sorted(posted)


_LOADER = {}  # in a Loader's process: the crypto context and what a fresh post looks like


def _loader_init(cc, dim, fresh):
    import openfhe as fhe
    _LOADER.update(cc=fhe.DeserializeCryptoContextString(cc, fhe.BINARY), dim=dim, fresh=tuple(fresh))


def _loads(blob):
    from threshold import chunks
    try:
        cts = chunks(blob, _LOADER["dim"])
        return all((x.GetLevel(), x.GetNoiseScaleDeg()) == _LOADER["fresh"] and _LOADER["cc"].EvalAdd(x, x) is not None
                   for x in cts)
    except Exception:
        return False


class Loader:
    """Loads posts in a separate process before the member does: OpenFHE can crash or hang (not raise) on hostile
    bytes, and in the member that would stop the run (post_crash_check.py). Spawned, not forked: a forked child of
    an OpenMP process can hang. If the process dies or hangs on a batch, each post is retried alone."""
    def __init__(self, cc, dim, fresh, timeout=120.0):
        self.init, self.timeout, self.pool = (cc, dim, fresh), timeout, None

    def _run(self, blobs):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        if self.pool is None:
            self.pool = ProcessPoolExecutor(1, multiprocessing.get_context("spawn"), _loader_init, self.init)
        try:
            return list(self.pool.map(_loads, blobs, timeout=self.timeout))
        except Exception:  # BrokenProcessPool (a crash) or TimeoutError (a hang)
            for p in list(self.pool._processes.values()):  # ponytail: private attr, the executor has no kill()
                p.kill()
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.pool = None
            return None

    def check(self, blobs):
        """[True if OpenFHE loads that post as a fresh-shaped ciphertext without crashing or hanging]"""
        got = self._run(blobs) if blobs else []
        return got if got is not None else [bool((r := self._run([b])) and r[0]) for b in blobs]


def invalid_posts(rnd, posts, me, fresh, loader=None):
    """Participants whose post every member drops instead of failing the run (gap_h.py): bad signature, or not
    ciphertexts shaped like a fresh encryption (count, size, level, scale degree: a lowered or extended one would
    pass loading and break scoring later). Without range proofs also the same bytes as another post, both dropped
    (who copied whom is unknowable, so a participant can knock out another by copying it); with range proofs the
    copier can't reuse the victim's proof, so PINE rejects the copier alone. The board keeps a participant's post
    write-once, so every member reads the same bytes and drops the same ones."""
    from collections import Counter as C

    from threshold import unpack
    copies, bad = C(blob for blob, _ in posts.values()), []
    for c, (blob, sig) in posts.items():
        try:
            ok = ((CFG.get("norm_bound") is not None or copies[blob] == 1)
                  and verify(CFG["participants"][c], f"post/{rnd}/{c}", blob, sig)
                  and [len(p) for p in unpack(blob)] == fresh[0])
        except Exception:
            ok = False
        bad += [c] * (not ok)
    left = [c for c in posts if c not in bad]
    if loader is not None:  # OpenFHE touches the bytes only once they've loaded safely elsewhere
        bad += [c for c, ok in zip(left, loader.check([posts[c][0] for c in left])) if not ok]
    for c in [c for c in left if c not in bad]:
        try:
            ok = all((x.GetLevel(), x.GetNoiseScaleDeg()) == fresh[1] for x in me._load(posts[c][0]))
        except Exception:
            ok = False
        bad += [c] * (not ok)
    return sorted(bad)


def check_roster(data):
    """Members only use a roster of distinct participant ids; [] means the lead skipped the round (too few posted)."""
    roster = json.loads(data)
    if (not isinstance(roster, list) or len(set(roster)) != len(roster) or 0 < len(roster) < CFG["min_participants"]
            or not all(isinstance(c, int) and 0 <= c < len(CFG["participants"]) for c in roster)):
        raise PermissionError(f"bad roster {roster!r}")
    return roster


def close_members(rnd, live):
    """Lead: the members still here, i.e. who posted this round's noise within member_timeout, plus members
    rejoining this round (a signed request and their noise already up; they aren't waited for)."""
    start = time.time()
    while True:
        here = [j for j in live if try_get(f"noise/{rnd}/{j}", "HEAD") is not None]
        if len(here) == len(live) or time.time() - start > CFG["member_timeout"]:
            back = [j for j in rejoining(rnd, live) if try_get(f"noise/{rnd}/{j}", "HEAD") is not None]
            return sorted(here + back)
        check_abort()
        time.sleep(0.05)


def rejoining(rnd, live):
    """Dropped members ranked after the lead with a signed rejoin request for this round."""
    out = []
    for j in range(min(live) + 1, len(CFG["committee"])):
        if j not in live and try_get(f"rejoin/{rnd}/{j}", "HEAD") is not None:
            mget(f"rejoin/{rnd}/{j}", j)  # raises unless j signed it
            out.append(j)
    return out


def check_live(data, live, rnd):
    """Members accept dropping members, adding back only members that asked to rejoin (signed), never going below
    the threshold, and never dropping the lead that posts this list (a gone lead is ruled on by its successor)."""
    new = json.loads(data)
    if (not isinstance(new, list) or new != sorted(set(new)) or not set(new) <= set(live) | set(rejoining(rnd, live))
            or len(new) < threshold(CFG) or min(live) not in new):
        raise PermissionError(f"bad member list {new!r} (live: {live})")
    return new


def seed(rnd, who):
    """Every training run is seeded by (round, participant), so the lead can replay it in plaintext."""
    return 1000 * rnd + who


def start_model(t):
    """Round 0's model: the config's (a resumed run) or the task's init."""
    import numpy as np
    w = t.init()
    if "init_model" not in CFG:
        return w
    m = np.frombuffer(base64.b64decode(CFG["init_model"]), dtype=np.float64).copy()
    if m.shape != w.shape:
        raise ValueError(f"the config's model has {len(m)} parameters, the task's {len(w)}")
    return m


def export(cfg_path, out):
    """Write the latest model on the board that t members signed: a new run starts from it (config --init-model)."""
    global CFG, SESSION
    with open(cfg_path) as f:
        CFG = json.load(f)
    SESSION = session(CFG)
    last = 0
    while try_get(f"model/{last + 1}", "HEAD") is not None:
        last += 1
    for rnd in range(last, 0, -1):  # the newest may lack signatures (its members died signing)
        model = get(f"model/{rnd}")
        signers = [j for j, m in enumerate(CFG["committee"]) if (sig := try_get(f"modelsig/{rnd}/{j}")) is not None
                   and verify(m["sign"], f"model/{rnd}", model, sig)]
        if len(signers) >= threshold(CFG):
            with open(out, "wb") as f:
                f.write(model)
            return print(f"{out}: model after round {rnd - 1}, signed by members {signers}")
    raise SystemExit("no model on the board is signed by enough members")


def task_name():
    return os.environ.get("TASK") or CFG["task"]


def task():
    """The task module: tasks/<name>.py, or the name itself if it's a .py path (contract in tasks/lr.py)."""
    import importlib.util
    name = task_name()
    builtin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks", f"{name}.py")  # next to this file
    spec = importlib.util.spec_from_file_location("task", name if name.endswith(".py") else builtin)
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    return t


def plain_step(t, w, rnd, roster):
    """One round of fl_core's defended plaintext FedAvg from model w, same participants and seeds.
    Checked round by round: over 30 CNN rounds a 1e-13 difference can flip a borderline
    accept decision, and the trajectories part ways."""
    import numpy as np

    from fl_core import keep
    ref = t.train(w, *t.root_data(), seed(rnd, 999))
    deltas = {c: (POISON_SCALE if c < POISON else 1) * t.train(w, *t.participant_data(c), seed(rnd, c)) for c in roster}
    if CFG.get("norm_bound") is not None:  # roster = who passed the range proof; they clipped to the bound
        from range_proof import clip
        deltas = {c: clip(d, CFG["norm_bound"]) for c, d in deltas.items()}
    ids = list(deltas)
    dots = [deltas[c] @ ref for c in ids]
    scale = {ids[i]: s for i, s in keep(dots, [deltas[c] @ deltas[c] for c in ids]).items()}
    kept, near = list(scale), min(map(abs, dots), default=np.inf)  # near: how close a keep decision came to flipping
    if len(kept) < 2:  # the protocol skips a round with < 2 kept (it won't reveal a single update)
        return w, kept, near
    return w + np.mean([s * deltas[c] for c, s in scale.items()], axis=0), kept, near


# --- committee member ---

def keygen(i, secret=None):
    """Member i's share of the chained key generation. Returns (cc, joint public key, my secret share).
    Signatures stop the board from swapping keys; they can't stop a member from generating its own share
    badly. ponytail: members are trusted to follow keygen; proofs of correct key generation if they aren't.
    secret: a rejoining member's own share (JSON, from its state file); then nothing is generated or posted."""
    import openfhe as fhe

    from threshold import ROTATIONS, context
    B, idx, last = fhe.BINARY, ROTATIONS, len(CFG["committee"]) - 1
    if secret is not None:
        cc = fhe.DeserializeCryptoContextString(mget("keygen/cc", 0), B)
        s = fhe.DeserializePrivateKeyString(secret, fhe.JSON)
    else:
        if i == 0:
            cc = context()
            sput("keygen/cc", fhe.Serialize(cc, B))
            kp = cc.KeyGen()
        else:
            cc = fhe.DeserializeCryptoContextString(mget("keygen/cc", 0), B)
            kp = cc.MultipartyKeyGen(fhe.DeserializePublicKeyString(mget(f"keygen/pk/{i - 1}", i - 1), B))
        sput(f"keygen/pk/{i}", fhe.Serialize(kp.publicKey, B))
        s, my_tag = kp.secretKey, kp.publicKey.GetKeyTag()
        if i == 0:  # relinearization and rotation keys: member 0 starts each chain
            base = cc.KeySwitchGen(s, s)
            cc.EvalAtIndexKeyGen(s, idx)
            rot = cc.GetEvalAutomorphismKeyMap(s.GetKeyTag())
            sput("keygen/relin/base", fhe.Serialize(base, B))
            sput("keygen/relin/0", fhe.Serialize(base, B))
            sput("keygen/rot/base", fhe.Serialize(rot, B))
            sput("keygen/rot/0", fhe.Serialize(rot, B))
        else:  # the others add their share to the running joint key
            base = fhe.DeserializeEvalKeyString(mget("keygen/relin/base", 0), B)
            prev = fhe.DeserializeEvalKeyString(mget(f"keygen/relin/{i - 1}", i - 1), B)
            sput(f"keygen/relin/{i}", fhe.Serialize(cc.MultiAddEvalKeys(prev, cc.MultiKeySwitchGen(s, s, base), my_tag), B))
            base_rot = fhe.DeserializeEvalKeyMapString(mget("keygen/rot/base", 0), B)
            prev_rot = fhe.DeserializeEvalKeyMapString(mget(f"keygen/rot/{i - 1}", i - 1), B)
            mine = cc.MultiEvalAtIndexKeyGen(s, base_rot, idx, my_tag)
            sput(f"keygen/rot/{i}", fhe.Serialize(cc.MultiAddEvalAutomorphismKeys(prev_rot, mine, my_tag), B))
    pk = fhe.DeserializePublicKeyString(mget(f"keygen/pk/{last}", last), B)
    if secret is None:
        relin = fhe.DeserializeEvalKeyString(mget(f"keygen/relin/{last}", last), B)
        sput(f"keygen/mult/{i}", fhe.Serialize(cc.MultiMultEvalKey(s, relin, pk.GetKeyTag()), B))
    mult = None
    for j in range(last + 1):
        part = fhe.DeserializeEvalKeyString(mget(f"keygen/mult/{j}", j), B)
        mult = part if mult is None else cc.MultiAddEvalMultKeys(mult, part, part.GetKeyTag())
    cc.InsertEvalMultKey([mult])
    cc.InsertEvalAutomorphismKey(fhe.DeserializeEvalKeyMapString(mget(f"keygen/rot/{last}", last), B))
    return cc, pk, s


@lru_cache(maxsize=None)
def pine_pool():
    """Worker processes for range-proof verification, one per core this member is given (spawned, not forked:
    OpenFHE's OpenMP runtime isn't fork-safe). Verifying is pure Python, ~7 s per participant."""
    import multiprocessing
    return multiprocessing.get_context("spawn").Pool(int(os.environ.get("OMP_NUM_THREADS") or os.cpu_count()))


def pine_round(i, rnd, me, ckey, roster, posts, live, reveal, dim, need):
    """Range proofs for this round's roster (range_proof.py): the participants that pass, in roster order."""
    import numpy as np
    import openfhe as fhe

    import range_proof as rp
    from threshold import pack, unpack
    B, bound = fhe.BINARY, CFG["norm_bound"]
    p = rp.pine(dim, bound, len(live))
    verify_key = hashlib.sha256(ckey + b"pine verify key").digest()[:p.VERIFY_KEY_SIZE]
    k = live.index(i)  # my PINE aggregator index: members in the order the lead listed them
    sealed = lambda name: unseal(ckey, name, get(name))
    # projections from every member's revealed randomness, each checked against its commitment
    sput(f"rreveal/{rnd}/{i}", reveal)
    reveals = [need(f"rreveal/{rnd}/{j}", j, lambda n, j=j: mget(n, j)) for j in live]
    if any(hashlib.sha256(r).digest() != mget(f"rcommit/{rnd}/{j}", j) for j, r in zip(live, reveals)):
        raise PermissionError("a member's revealed randomness doesn't match its commitment")
    seed = hashlib.sha256(SESSION + b"".join(posts[c][0] for c in roster) + b"".join(reveals)).digest()
    rs = np.random.default_rng(int.from_bytes(seed, "big")).choice([-1.0, 1.0], (rp.CHECKS, dim))
    jobs = []
    for c in roster:
        try:
            nonce, public, *shares = unpack(sget(f"proof/{rnd}/{c}", CFG["participants"][c]))
            mine = from_participant(f"proof/{rnd}/{c}/{i}", shares[k]) if len(shares) == len(live) else None
            jobs.append((dim, bound, len(live), verify_key, k, nonce, public, mine) if mine is not None else None)
        except Exception as e:  # malformed or not sealed to me: this participant fails, the round goes on
            print(f"member {i} round {rnd}: participant {c}'s proof unreadable: {type(e).__name__}: {e}", file=sys.stderr)
            jobs.append(None)
    done = iter(pine_pool().map(rp.prep_job, [j for j in jobs if j is not None]))
    started = [next(done) if j is not None else None for j in jobs]
    name = f"pine/{rnd}/{i}"
    proj = me.projections({c: posts[c] for c in roster}, rs)
    put(name, seal(ckey, name, pack(pack(*(s[1] if s else b"" for s in started)),
                                    pack(*(fhe.Serialize(x, B) for x in proj)))))
    theirs = [[unpack(part) for part in unpack(need(f"pine/{rnd}/{j}", j, sealed))] for j in live]
    mine_proj = []
    for n, (c, s) in enumerate(zip(roster, started)):
        preps = [t[0][n] for t in theirs]
        out = rp.finish(p, s[0], preps) if s and all(preps) else None
        mine_proj.append(json.dumps(rp.project(p, out, rs)).encode() if out is not None else b"")
    name = f"pineout/{rnd}/{i}"
    put(name, seal(ckey, name, pack(*mine_proj)))
    outs = [unpack(need(f"pineout/{rnd}/{j}", j, sealed)) for j in live]
    from_ct = me.read_projections([[fhe.DeserializeCiphertextString(b, B) for b in t[1]] for t in theirs],
                                  len(roster), rp.CHECKS)
    tol, passing = rp.tolerance(len(live), bound, dim), []
    for n, c in enumerate(roster):
        if not all(o[n] for o in outs):
            print(f"member {i} round {rnd}: participant {c} failed PINE", file=sys.stderr)
            continue
        gap = np.max(np.abs(from_ct[n] - rp.combine(p, [json.loads(o[n]) for o in outs])))
        if gap <= tol:  # NaN (didn't decode) fails
            passing.append(c)
        else:
            print(f"member {i} round {rnd}: participant {c}'s ciphertext differs from its shares by {gap:.1e}",
                  file=sys.stderr)
    return passing


def member(i):
    import numpy as np
    import openfhe as fhe

    from threshold import Member, decrypt_std, n_chunks, pack, share_key, unpack
    B, lead, n_members, n_part = fhe.BINARY, i == 0, len(CFG["committee"]), len(CFG["participants"])
    t0 = time.time()
    state = f"{KEY_PATH}.state"  # my key share and the committee key, so I can come back after a crash
    saved = json.load(open(state)) if os.path.exists(state) else None
    back = saved is not None and saved["session"] == SESSION.hex()
    cc, pk, share = keygen(i, saved["share"] if back else None)
    ckey = bytes.fromhex(saved["ckey"]) if back else committee_key(i)
    n_t = threshold(CFG)
    shape = saved["shape"] if back else None
    if n_t < n_members and not back:  # t-of-n: Shamir-share my key, so any t members can stand in for me
        mine = share_key(share, n_members, n_t)
        for j in range(n_members):
            if j != i:
                put(f"keyshare/{i}/{j}", seal(pairwise(j), f"keyshare/{i}/{j}", mine[j].tobytes()))
        shape = list(mine[i].shape)
    if not back:  # owner-only, like the .key file; ponytail: plaintext at rest, encrypt it if the disk isn't trusted
        with os.fdopen(os.open(state, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
            json.dump({"session": SESSION.hex(), "share": fhe.Serialize(share, fhe.JSON), "ckey": ckey.hex(),
                       "shape": shape}, f)
    held = lambda j: np.frombuffer(unseal(pairwise(j), f"keyshare/{j}/{i}", get(f"keyshare/{j}/{i}")),
                                   dtype=np.uint64).reshape(shape)  # my share of member j's key
    t_keygen = time.time() - t0
    t = task()
    w0, rounds = start_model(t), t.ROUNDS
    dim = len(w0)
    authentic = [lambda rnd, blob, tag, c=c: verify(CFG["participants"][c], f"post/{rnd}/{c}", blob, tag) for c in range(n_part)]
    sigma = CFG.get("dp_z", 0) * (CFG.get("norm_bound") or 0)  # std of each member's noise on each round's sum
    me = Member(cc, pk, share, lead, authentic, t.root_data(), w0, sigma, seed=i,
                train=lambda w, X, y, rnd: t.train(w, X, y, seed(rnd, 999)),
                committee=n_members, min_clients=CFG["min_participants"], bound=CFG.get("norm_bound"))
    bounded = CFG.get("norm_bound") is not None
    if lead and not back:  # the members of each round (rejoining members, range-proof shares)
        sput("members/0", json.dumps(list(range(n_members))).encode())
    ct = lambda b: fhe.DeserializeCiphertextString(b, B)
    sealed = lambda name: unseal(ckey, name, get(name))
    round_s, rosters, rosters_all, honest_rejected, skipped, fallbacks, history, kepts = [], [], [], 0, 0, 0, [w0], []
    live, member_drops, unproven, member_rejoins, first = list(range(n_members)), {}, {}, {}, 0
    dropped_posts, empty = {}, 0  # round: participants whose post was invalid; rounds in a row with no roster
    slow = {}  # round: seconds per phase, for rounds over a minute (finding intermittent stalls)
    from threshold import chunks as split, encrypt as enc, unpack as parts
    probe = enc(cc, pk, np.zeros(dim))  # what a fresh post looks like
    fresh = ([len(p) for p in parts(probe)], (split(probe, dim)[0].GetLevel(), split(probe, dim)[0].GetNoiseScaleDeg()))
    loader = Loader(fhe.Serialize(cc, B), dim, fresh[1])
    if back:  # the latest round whose member list is out; the model of it only if t members signed it
        while try_get(f"members/{first + 1}", "HEAD") is not None:
            first += 1
        live = json.loads(from_members(f"members/{first}"))
        if first:
            model = get(f"model/{first}")
            signers = [j for j in live if (sig := try_get(f"modelsig/{first}/{j}")) is not None
                       and verify(CFG["committee"][j]["sign"], f"model/{first}", model, sig)]
            if len(signers) < n_t:
                raise PermissionError(f"model {first} isn't signed by {n_t} members")
            me.w = np.frombuffer(model, dtype=np.float64).copy()
        me.rnd, lead = first, False
        print(f"member {i}: rejoining at round {first}", file=sys.stderr)
    crashed, mid = crashed_members(n_members), CRASH.endswith("-mid")
    crash = lambda rnd, now: (SIMULATE and not back and i in crashed and n_t < n_members and rnd == rounds // 2
                              and now == mid)

    def heartbeat():  # what deciders watch; a crashed process stops beating
        while True:
            try:
                sput(f"beat/{i}", str(time.time_ns()).encode())
            except Exception:
                pass
            time.sleep(0.5)
    threading.Thread(target=heartbeat, daemon=True).start()

    def need(name, j, read=get, excluded=frozenset()):
        """Member j's message `name`, or Gone. The lowest live member other than j rules whether it arrived or j's
        heartbeat stopped first; that ruling is itself a message, ruled on the same way by the next lowest member, so
        a decider that dies before ruling is caught (the successor alone, or the lead with it). Everyone, j too,
        follows the second ruling, read directly while watching its author's heartbeat for 3 member timeouts. A
        strict chain always ends in someone whose death hangs it (before 2026-10-12: the successor's, alone); this
        ends in a local timeout instead, so only a second decider frozen that long yet alive could split members."""
        others = sorted(set(live) - {j} - excluded)
        if len(excluded) == 2 or not others:  # no one left to rule: watch j itself
            if not arrived(name, j, 3 * CFG["member_timeout"]):
                raise Gone(j)
            return read(name)
        decider, vname = others[0], f"verdict/{name}"
        if i == decider:
            sput(vname, b"ok" if arrived(name, j) else b"gone")
        if need(vname, decider, lambda n: mget(n, decider), excluded | {j}) == b"gone":
            raise Gone(j)
        return read(name)

    for rnd in range(first, rounds):
        if i not in live:  # rejoining: ask in for this round
            sput(f"rejoin/{rnd}/{i}", b"")
        if crash(rnd, False):
            print(f"member {i}: simulated crash in round {rnd}", file=sys.stderr)
            return
        t_round, roster = time.time(), []
        marks = [("start", t_round)]
        mark = lambda name: marks.append((name, time.time()))
        try:
            leader = min(live)
            if lead:
                put(f"roster/{rnd}", seal(ckey, f"roster/{rnd}", json.dumps(close_round(rnd)).encode()))
            roster = check_roster(need(f"roster/{rnd}", leader, sealed))
            mark("roster")
            posts = {c: (p[:-SIG], p[-SIG:]) for c in roster for p in [get(f"post/{rnd}/{c}")]}
            if bad := invalid_posts(rnd, posts, me, fresh, loader):  # too few left: the round is skipped below
                dropped_posts[rnd] = bad
                roster = [c for c in roster if c not in bad]
                posts = {c: posts[c] for c in roster}
            if bounded:  # commit to my share of the projection randomness before anyone reveals theirs
                reveal = os.urandom(32)
                sput(f"rcommit/{rnd}/{i}", hashlib.sha256(reveal).digest())
            sput(f"noise/{rnd}/{i}", me.noise(dim, split=False))
            if lead:
                put(f"live/{rnd}", seal(ckey, f"live/{rnd}", json.dumps(close_members(rnd, live)).encode()))
            new = check_live(need(f"live/{rnd}", leader, sealed), live, rnd)
            mark("live")
            if i not in new:
                if i not in live:  # not admitted yet: follow the run to the next round's model and ask again
                    published = need(f"model/{rnd + 1}", leader)
                    me.abandon(rnd)
                    me.w = np.frombuffer(published, dtype=np.float64).copy()
                    continue
                raise SystemExit(f"member {i}: counted as dropped in round {rnd}")
            if new != live:  # my partial decryptions now stand in for exactly the members not in it
                if set(live) - set(new):
                    member_drops.setdefault(rnd, []).extend(sorted(set(live) - set(new)))
                if set(new) - set(live):
                    member_rejoins.setdefault(rnd, []).extend(sorted(set(new) - set(live)))
                live = new
                me.recover(i, [held(j) for j in range(n_members) if j not in live], live)
            noises = [mget(f"noise/{rnd}/{j}", j) for j in live]
            mark("noise")
            empty = empty + 1 if not roster else 0
            if empty >= 3:  # the same on every member: they all read the lead's rosters
                raise PermissionError(f"{empty} rounds in a row without {CFG['min_participants']} posts: stopping")
            scored = roster
            if bounded and len(roster) >= CFG["min_participants"]:
                scored = None  # the participants split their shares among other members: can't be checked
                if json.loads(from_members(f"members/{rnd}")) == live:
                    scored = pine_round(i, rnd, me, ckey, roster, posts, live, reveal, dim, need)
                    unproven[rnd] = sorted(set(roster) - set(scored))
                    mark("proofs")
            if scored is None or len(scored) < CFG["min_participants"]:  # members changed, or too few proven
                me.skip(rnd)
                partial, kept, scored = [], [], []
            else:
                posts = {c: posts[c] for c in scored}
                name = f"score/{rnd}/{i}"
                put(name, seal(ckey, name, pack(*(fhe.Serialize(p, B) for p in me.scores(rnd, posts, noises)))))
                if crash(rnd, True):
                    print(f"member {i}: simulated crash mid-round {rnd}", file=sys.stderr)
                    return
                score_partials = [[ct(b) for b in unpack(need(f"score/{rnd}/{j}", j, sealed))] for j in live]
                scores = me.read_scores(score_partials)
                name = f"fallback/{rnd}/{i}"  # sealed, so the board can't forge a member's vote
                put(name, seal(ckey, name, b"1" if scores is None else b"0"))
                if any(need(f"fallback/{rnd}/{j}", j, sealed) == b"1" for j in live):
                    fallbacks += 1
                    name = f"each/{rnd}/{i}"
                    put(name, seal(ckey, name, pack(*(fhe.Serialize(p, B) for pair in me.scores_each() for p in pair))))
                    each = []
                    for j in live:
                        bs = [ct(b) for b in unpack(need(f"each/{rnd}/{j}", j, sealed))]
                        if len(bs) != 2 * len(posts):
                            raise PermissionError(f"member {j}'s fallback scores have the wrong length")
                        each.append(list(zip(bs[::2], bs[1::2])))
                    scores = me.read_each(each)
                partial, kept = me.aggregate(*scores)
                mark("scores")
            sput(f"agg/{rnd}/{i}", pack(*(fhe.Serialize(p, B) for p in partial)))
            partials = [[ct(b) for b in unpack(need(f"agg/{rnd}/{j}", j, lambda n, j=j: mget(n, j)))] for j in live]
            mark("agg")
            if any(len(p) != (n_chunks(dim) if len(kept) >= 2 else 0) for p in partials):
                raise PermissionError("a member's aggregate has the wrong number of chunks")
            if lead:
                w = me.update(partials)
                sput(f"members/{rnd + 1}", json.dumps(live).encode())  # before the model: participants split by it
                put(f"model/{rnd + 1}", w.tobytes())
            published = need(f"model/{rnd + 1}", leader)
            mark("model")
            if not lead:
                w = me.update(partials, np.frombuffer(published, dtype=np.float64))
        except Gone as gone:  # abandon this round: drop the member, keep the model
            j = gone.args[0]
            mark(f"gone {j}")
            if j == i:  # SystemExit: the others go on without it, so no abort
                raise SystemExit(f"member {i}: ruled gone in round {rnd}")
            # RuntimeError, not SystemExit: run_role then posts the abort, so participants stop now instead of
            # waiting 30 min for a model no one can decrypt any more (before 2026-10-15)
            if n_t == n_members:
                raise RuntimeError(f"member {j} is gone and every member is needed (threshold n of n)")
            member_drops.setdefault(rnd, []).append(j)
            live = sorted(set(live) - {j})
            if len(live) < n_t:
                raise RuntimeError(f"fewer than {n_t} members left, the run can't decrypt any more")
            me.recover(i, [held(k) for k in range(n_members) if k not in live], live)
            lead = i == min(live)
            me.abandon(rnd)
            w, kept, scored = me.w.copy(), [], []
            if lead:  # model/rnd too, if missing: last round was abandoned while its lead was dead (unnoticed, its
                # ruler died with it), so no one posted it; the model hasn't changed since, and members signed that
                for r in [rnd] * (rnd > 0 and try_get(f"model/{rnd}", "HEAD") is None) + [rnd + 1]:
                    sput(f"members/{r}", json.dumps(live).encode())
                    put(f"model/{r}", w.tobytes())
        # participants train on a model only if t members signed it (every member, if t = n)
        put(f"modelsig/{rnd + 1}/{i}", ME[0].sign(signed(f"model/{rnd + 1}", w.tobytes())))
        honest_rejected += sum(c not in kept for c in scored if c >= POISON)  # scored = roster without proofs
        skipped += len(kept) < 2
        rosters.append(scored)
        rosters_all.append(roster)
        history.append(w.copy())
        kepts.append(kept)
        round_s.append(time.time() - t_round)
        if round_s[-1] > 60:
            slow[rnd] = {b[0]: round(b[1] - a[1], 1) for a, b in zip(marks, marks[1:] + [("end", time.time())])}
            print(f"member {i}: slow round {rnd}: {slow[rnd]}", file=sys.stderr, flush=True)
    if not lead:
        return
    try:  # what the board holds for round 0's scores is useless to it
        [fhe.DeserializeCiphertextString(b, B) for b in unpack(get("score/0/0"))]
        board_reads = "YES"
    except Exception:
        board_reads = "no (sealed)"
    stats = json.loads(get("stats"))
    print(f"task {task_name()}: {dim} parameters = {n_chunks(dim)} ciphertext(s) per update")
    print(f"{n_members} members, {n_part} participants ({POISON} poisoning), {rounds} rounds, each a separate container\n")
    if "init_model" in CFG and hasattr(t, "evaluate"):
        print(f"started from the config's model: {t.evaluate(w0)}")
    print(f"deployed: {t.evaluate(w) if hasattr(t, 'evaluate') else 'done (task has no evaluate)'}")
    if SIMULATE:
        steps = [plain_step(t, history[r], r, rosters[r]) if rosters[r] else (history[r], [], np.inf) for r in range(rounds)]
        same = [r for r, (_, k, _) in enumerate(steps) if k == kepts[r]]
        err = max([np.abs(steps[r][0] - history[r + 1]).max() for r in same], default=0.0)
        kept_diff = rounds - len(same)
        tol = 8 * (decrypt_std(n_members) + sigma * np.sqrt(n_members) / 2)  # every member's noise, over >= 2 kept
        print(f"each round vs one plaintext defended round from the same model: max |w - plain| {err:.1e} "
              f"(flooding std {decrypt_std(n_members):.1e}, noise on the sum {sigma:.1e} per member), keep/reject decisions differing in {kept_diff} rounds "
              f"(a score within the flooding of a threshold)")
    print(f"honest rejected {honest_rejected}, rounds skipped (< 2 kept) {skipped}, "
          f"rounds scored one by one (packed scores untrustworthy) {fallbacks}")
    print(f"board can parse the score partials it relays: {board_reads}")
    short = {r: n_part - len(ro) for r, ro in enumerate(rosters_all) if len(ro) < n_part}
    print(f"rounds closed without everyone (round: missing): {short}")
    print(f"members dropped (round: members), {n_t} of {n_members} needed to decrypt: {member_drops}"
          + (f"; rejoined: {member_rejoins}" if member_rejoins else ""))
    if bounded:
        bad = {r: c for r, c in unproven.items() if c}
        print(f"range proofs |u| <= {CFG['norm_bound']}: failed (round: participants) {bad}; rounds not scored "
              f"(member set changed or < {CFG['min_participants']} proven): {sum(1 for r in rosters if not r)}")
    print(f"key generation {t_keygen:.1f}s, round {np.median(round_s):.1f}s median ({min(round_s):.1f}..{max(round_s):.1f})\n")
    if slow:
        print(f"rounds over a minute (seconds per phase, each up to that mark): {slow}\n")
    print(f"{'traffic via the board':22} {'posted MB':>10} {'fetched MB':>11} {'per round MB':>13}")
    for kind in ("keygen", "keyshare", "post", "proof", "noise", "pine", "pineout", "score", "fallback", "each",
                 "agg", "model"):
        up, down = stats["up"].get(kind, 0) / 1e6, stats["down"].get(kind, 0) / 1e6
        if kind in ("proof", "pine", "pineout") and not up:  # range proofs off
            continue
        per = f"{(up + down) / rounds:13.1f}" if kind not in ("keygen", "keyshare") else f"{'-':>13}"
        print(f"{kind:22} {up:10.1f} {down:11.1f} {per}")
    if SIMULATE:
        # a keep decision may differ only where a plaintext score u.ref is within the decryption flooding of 0: a
        # converged model (a resumed run, INIT_MODEL) has honest scores ~1e-4..2e-3 against flooding std ~4e-4, and
        # flipped 12 of 20 rounds, past the old "a quarter of rounds" allowance (2026-10-15)
        flips = [r for r in range(rounds) if r not in same]
        assert err < tol and all(steps[r][2] < 8 * decrypt_std(n_members) for r in flips), \
            f"deployed rounds must match plaintext defended (flips {[(r, steps[r][2]) for r in flips]})"
        assert not sigma or not same or err > sigma / n_part, "the members' noise must be in the published model"
    assert board_reads != "YES", "the board must not be able to fuse the scores"
    if SIMULATE:  # the crash round may be skipped; 21 of 30 were when the deadline ran from the round's start
        assert skipped <= rounds // 10, f"most rounds must train, {skipped} of {rounds} skipped"
    print(f"invalid posts dropped (round: participants): {dropped_posts}")
    if SIMULATE:
        want = ({2: [3], 6: [3]} if bounded else {2: [3], 4: [3, 4], 6: [3]}) if BAD_POST else {}
        if TAMPER and TAMPER.startswith("post/"):  # a tampered post fails its signature: dropped, not fatal
            want[int(TAMPER.split("/")[1])] = [int(TAMPER.split("/")[2])]
        assert all(dropped_posts.get(r, v) == v for r, v in want.items()) and set(dropped_posts) <= set(want), \
            f"members must drop exactly the invalid posts, got {dropped_posts}"
    if SIMULATE and n_t < n_members:
        assert sorted(sum(member_drops.values(), [])) == crashed and min(member_drops) == rounds // 2 \
            and max(member_drops) <= rounds // 2 + 1, f"the run must go on without the crashed members, got {member_drops}"
        if "rejoin" in CRASH:
            assert list(member_rejoins.values()) == [crashed] and min(member_rejoins) > rounds // 2, \
                f"the crashed member must rejoin, got {member_rejoins}"
    if SIMULATE and bounded:  # participant 0 sends its update unclipped with a lying proof; everyone else clips
        too_big = {r: bool(np.linalg.norm(POISON_SCALE * t.train(history[r], *t.participant_data(0), seed(r, 0)))
                           > CFG["norm_bound"]) for r in unproven if 0 in rosters_all[r]}
        print(f"participant 0's update over the bound in rounds {[r for r, b in too_big.items() if b]} "
              f"(within it once training converges, and then its proof is true)")
        copier = {4: [3]} if BAD_POST else {}  # participant 3's copy of 4's post can't carry a proof of it
        assert all(c == sorted(([0] if too_big.get(r) else []) + copier.get(r, [])) for r, c in unproven.items()), \
            f"the range proofs must reject exactly the out-of-bound liar, got {unproven}"


def joint_key():
    """The committee's crypto context and joint public key, each signed by its maker."""
    import openfhe as fhe
    B, last = fhe.BINARY, len(CFG["committee"]) - 1
    cc = fhe.DeserializeCryptoContextString(mget("keygen/cc", 0), B)
    return cc, fhe.DeserializePublicKeyString(mget(f"keygen/pk/{last}", last), B)  # the joint key, signed by its maker


def signed_model(rnd):
    """Round rnd's model (rnd >= 1) once t members signed it: whoever serves it can't hand out a model of its own."""
    import numpy as np
    data = get(f"model/{rnd}")
    signers = set()  # dropped members never sign, so wait for t signatures, not all
    while len(signers) < threshold(CFG):
        for j, m in enumerate(CFG["committee"]):
            if j not in signers and (sig := try_get(f"modelsig/{rnd}/{j}")) is not None:
                if not verify(m["sign"], f"model/{rnd}", data, sig):
                    raise PermissionError(f"model {rnd}: bad signature from member {j}")
                signers.add(j)
        check_abort()
        time.sleep(0.05)
    return np.frombuffer(data, dtype=np.float64)


def prove(c, rnd, delta, lie=False):
    """With a norm bound: clip the update and post its range proof, which goes up before the post (the lead counts a
    post once both are in). Returns the update to encrypt. lie: the demo's liar, unclipped with a lying proof."""
    if CFG.get("norm_bound") is None:
        return delta
    import range_proof as rp
    from threshold import pack
    bound, members = CFG["norm_bound"], json.loads(from_members(f"members/{rnd}"))
    p = rp.pine(len(delta), bound, len(members))
    delta = delta if lie else rp.clip(delta, bound)
    nonce, public, shares = rp.prove(rp.liar(p) if lie else p, delta)
    name = f"proof/{rnd}/{c}"
    sput(name, pack(nonce, public, *(to_member(j, f"{name}/{j}", sh) for j, sh in zip(members, shares))))
    return delta


def participant(c):
    from threshold import encrypt
    cc, pk = joint_key()
    t = task()
    w, rounds = start_model(t), t.ROUNDS
    X, y = t.participant_data(c)
    offline = {int(r) for r in os.environ.get("OFFLINE", "").split(",") if r}
    for rnd in range(rounds):
        if rnd:
            w = signed_model(rnd)
        if rnd in offline or (WITHHOLD and SIMULATE and (c < POISON and rnd >= 5 or c == 7 and rnd >= 15)):
            continue
        delta = t.train(w, X, y, seed(rnd, c))
        if c < POISON:
            delta = POISON_SCALE * delta  # default -10: same poisoning as fl_core
        delta = prove(c, rnd, delta, lie=SIMULATE and c == 0 and POISON)  # the demo's liar: participant 0
        post = encrypt(cc, pk, delta)
        if BAD_POST and SIMULATE and c == 3 and rnd == 2:
            post = b"not a ciphertext"
        if BAD_POST and SIMULATE and c == 3 and rnd == 4:  # a copy of participant 4's post, signed as mine
            post = get(f"post/{rnd}/4")[:-SIG]
        if BAD_POST and SIMULATE and c == 3 and rnd == 6:  # right size and shape, but OpenFHE crashes loading it
            from post_crash_check import hostile
            post = hostile(post)
        sput(f"post/{rnd}/{c}", post)


# --- demo launcher (host) ---

def docker(*args, check=True):
    return subprocess.run(["docker", *args], check=check, capture_output=True, text=True).stdout.strip()


def self_signed(run, host):
    """Throwaway certificate for the demo board (a real board gets one from a CA or a reverse proxy)."""
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, host)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now).not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(host)]), critical=False).sign(key, hashes.SHA256()))
    with open(f"{run}/cert.pem", "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(f"{run}/key.pem", "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))


def resume_demo():
    """RESUME=1: the lead and its successor crash together (3 members: too few left to decrypt, the run aborts),
    then the run goes on as a new one from its last signed model (export, config --init-model)."""
    first = subprocess.run([sys.executable, __file__], capture_output=True, text=True,
                           env=os.environ | {"RESUME": "", "CRASH": "lead+successor", "EXPORT": "resume.bin"})
    print(first.stdout[-3000:])
    try:
        assert first.returncode == 1 and "fewer than 2 members left" in first.stdout, "the first run must abort"
        assert "signed by members" in first.stdout, "the export must find a signed model"
        sys.exit(subprocess.run([sys.executable, __file__], env=os.environ | {"RESUME": "", "INIT_MODEL": "resume.bin"})
                 .returncode)
    finally:
        if os.path.exists("resume.bin"):
            os.remove("resume.bin")


def launch():
    n_members = int(os.environ.get("MEMBERS", 3))  # MEMBERS=4 THRESHOLD=2: room for two members to crash
    # REMOTE_MEMBER=1: member 1 runs on another machine (its m1.pub in this folder); the board is published on
    # this host's port, and everyone reaches it as BOARD_HOST: host.docker.internal for a WSL VM on this laptop,
    # or this laptop's Tailscale name for a friend's computer (friend/README.md)
    remote = int(os.environ["REMOTE_MEMBER"]) if os.environ.get("REMOTE_MEMBER") else None
    if remote is not None and not (0 < remote < n_members and os.path.exists(f"m{remote}.pub")):
        raise SystemExit(f"REMOTE_MEMBER must be 1..{n_members - 1} (not the lead) with its m{remote}.pub in this folder")
    local = [i for i in range(n_members) if i != remote]
    members = {i: f"concilium-member-{i}" for i in local}
    flower = os.environ.get("FLOWER") in ("1", "deploy")  # the participants as Flower nodes instead (flower_check.py)
    names = ["concilium-board", *members.values()] + (["concilium-flower"] if flower else
                                                     [f"concilium-client-{c}" for c in range(N_CLIENTS)])
    run = ".concilium-demo"  # inside this folder: the containers see it as /work/.concilium-demo
    shutil.rmtree(run, ignore_errors=True)
    os.makedirs(run)
    for p in [f"m{i}" for i in local] + [f"p{c}" for c in range(N_CLIENTS)]:
        keys(f"{run}/{p}")
    if remote is not None:
        shutil.copy(f"m{remote}.pub", f"{run}/m{remote}.pub")
    host = "concilium-board" if remote is None else os.environ.get("BOARD_HOST", "host.docker.internal")
    tls = os.environ.get("TLS") == "1"
    if tls:
        self_signed(run, host)
    config([f"{run}/concilium.json", "--board", f"{'https' if tls else 'http'}://{host}:{PORT}",
            "--task", os.environ.get("TASK", "lr"), "--committee", *[f"{run}/m{i}.pub" for i in range(n_members)],
            # a remote member fetches each round's posts through a home upload: give it time
            "--participants", *[f"{run}/p{c}.pub" for c in range(N_CLIENTS)],
            "--member-timeout", "10" if remote is None else "120",
            *(["--threshold", os.environ["THRESHOLD"]] if os.environ.get("THRESHOLD") else []),
            *(["--round-timeout", "20"] if WITHHOLD else []),  # the demo shouldn't wait 5 min per skipped round
            *(["--init-model", os.environ["INIT_MODEL"]] if os.environ.get("INIT_MODEL") else []),
            *(["--dp-z", os.environ["DP_Z"]] if os.environ.get("DP_Z") else []),
            *(["--norm-bound", NORM_BOUND] if float(NORM_BOUND) else ["--no-norm-bound"])])
    if remote is not None:
        print(f"now start member {remote} on the other machine with {run}/concilium.json"
              f"{f' and SSL_CERT_FILE={run}/cert.pem' if tls else ''} (30 min before the others give up)", flush=True)
    base = ["run", "-d", "--network", NET, "-v", f"{os.getcwd()}:/work", "-e", f"SIMULATE={int(SIMULATE)}", "-e", f"POISON_SCALE={POISON_SCALE}", "-e", f"TAMPER={TAMPER or ''}", "-e", f"BAD_POST={int(BAD_POST)}", "-e", f"REPLAY_BEAT={int(REPLAY_BEAT)}", "-e", f"WITHHOLD={int(WITHHOLD)}",
            "-e", f"CRASH={CRASH}"]
    if tls:
        base += ["-e", f"SSL_CERT_FILE=/work/{run}/cert.pem"]
    # one host: split the cores between the members; OpenMP's default (all cores each) oversubscribes
    threads = max(1, (os.cpu_count() or n_members) // n_members)
    docker("rm", "-f", *names, check=False)
    docker("network", "create", NET, check=False)
    t = time.time()
    try:
        docker(*base, "--name", names[0], *(["-p", f"{PORT}:{PORT}"] if remote is not None else []),
               IMAGE, "python3", "deploy.py", "board", f"{run}/concilium.json",
               *([f"{run}/cert.pem", f"{run}/key.pem"] if tls else []))
        if remote is not None:  # local containers reach the board directly, not through Docker's port forwarder
            ip = docker("inspect", "-f", f'{{{{(index .NetworkSettings.Networks "{NET}").IPAddress}}}}', names[0])
            base += ["--add-host", f"{host}:{ip}"]
        for i, name in members.items():
            docker(*base, "--name", name, "-e", f"OMP_NUM_THREADS={threads}",
                   "-e", "OMP_WAIT_POLICY=PASSIVE",  # idle threads sleep: ~half the CPU, ~9% slower HE
                   IMAGE, "python3", "deploy.py", "member", f"{run}/concilium.json", f"{run}/m{i}.key")
        if flower:  # the Flower client-node image; the simulation adds Ray (unpinned: only this check needs it)
            subprocess.run(["docker", "build", "--target", "flower", "-t", "concilium-flower", "."], check=True)
            image = "concilium-flower"
            if os.environ["FLOWER"] != "deploy":
                image = "concilium-flower-sim"
                subprocess.run(["docker", "build", "-t", image, "-"], check=True, text=True,
                               input="FROM concilium-flower\nRUN pip3 install --no-cache-dir --break-system-packages "
                                     "ray==2.55.1\n")
            docker(*base, "--name", "concilium-flower", "-e", "OMP_NUM_THREADS=1", "-e", "PYTHONPATH=/work",
                   "-e", "PINE_MALICIOUS=1",  # participant 0's own proof checks off (honest ones' proofs still verify)
                   "--shm-size", "1g", image, "python3", "flower_check.py",
                   *(["deploy"] if os.environ["FLOWER"] == "deploy" else []))
        for c in [] if flower else range(N_CLIENTS):
            docker(*base, "--name", f"concilium-client-{c}", "-e", "OMP_NUM_THREADS=1",
                   "-e", f"PINE_MALICIOUS={int(bool(float(NORM_BOUND)) and SIMULATE and c == 0)}",  # the liar's own checks off
                   "-e", "OFFLINE=" + ",".join(map(str, OFFLINE.get(c, []))),
                   IMAGE, "python3", "deploy.py", "participant", f"{run}/concilium.json", f"{run}/p{c}.key")
        if SIMULATE and "rejoin" in CRASH:  # restart the crashed member: it finds its state file and rejoins
            name = members[crashed_members(n_members)[0]]
            docker("wait", name)
            # past 3x the demo's 10 s member timeout (need()'s last watch), so it's dropped first (sooner, it carries on)
            time.sleep(45)
            docker("start", name)
        codes = {n: int(docker("wait", n)) for n in members.values()}
        if not any(codes.values()):  # participants only finish once the members do; don't wait on them otherwise
            codes |= {n: int(docker("wait", n)) for n in names[1 + len(members):]}
        else:  # but say why those that quit did
            codes |= {n: int(docker("inspect", "-f", "{{.State.ExitCode}}", n)) for n in names[1 + len(members):]
                      if docker("inspect", "-f", "{{.State.Status}}", n, check=False) == "exited"}
        logs = [docker("logs", m) for m in members.values()]
        print(next((log for log in logs if "deployed:" in log), logs[0]))  # the lead at the end prints the report
        failed = [n for n, code in codes.items() if code]
        for n in failed + names[:1] * bool(failed):  # and the board's refusals
            print(f"\n--- {n} exited {codes.get(n, '-')} ---\n" + subprocess.run(["docker", "logs", n], capture_output=True,
                                                                          text=True).stderr[-3000:])
        print(f"\nwall time {time.time() - t:.0f}s")
        if os.environ.get("EXPORT"):  # the run's last signed model, for INIT_MODEL= in the next run
            print(docker("run", "--rm", *base[2:], IMAGE, "python3", "deploy.py", "export", f"{run}/concilium.json",
                         os.environ["EXPORT"]))
        sys.exit(1 if failed else 0)
    finally:
        docker("rm", "-f", *names, check=False)
        docker("network", "rm", NET, check=False)
        shutil.rmtree(run, ignore_errors=True)


if __name__ == "__main__":
    role, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("launch", [])
    if role == "board":
        CFG = json.load(open(args[0]))  # who may write what (may_write)
        check_config(CFG)
        SESSION = session(CFG)
        board_server(*args[1:]).serve_forever()
    elif role == "version":  # compare with the other parties (the session id covers it too)
        print(f"concilium code {code_hash().hex()[:16]}")
    elif role == "keys":
        keys(args[0])
        print(f"{args[0]}.key: your private keys, keep it here. {args[0]}.pub: send it to whoever writes the config.")
    elif role == "config":
        config(args)
    elif role == "export" and len(args) == 2:
        export(*args)
    elif role == "member":
        run_role(member, role, args)
    elif role == "participant":
        run_role(participant, role, args)
    elif role == "launch" and not args:
        resume_demo() if os.environ.get("RESUME") == "1" else launch()
    else:  # --help, typos: don't start the local demo
        sys.exit(__doc__[__doc__.index("Real deployment"):__doc__.index("Local demo")].rstrip())
