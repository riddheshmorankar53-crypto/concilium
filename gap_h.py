"""Gap H: one participant can stop the whole run (availability), found overnight 2026-10-10.

The board is untrusted and stores whatever anyone PUTs. deploy.py signs every protocol message, so a write
by the wrong party is caught, but the reader's reaction to a bad message is to post the global abort. And
the abort itself is unsigned ("the board can stop a run anyway"). Both assume only the board can write.
Any participant can write too, so one participant has the board's power to stop the run:
    abort      PUT abort/<session> ("participant 3: ...")    -> everyone stops at their next check
    overwrite  PUT junk over another participant's post      -> members see a bad signature, abort
Verified 2026-10-11 on the LR demo: abort stopped all 13 parties 18 s in; overwrite made the members raise
"bad MAC/signature for client 4" and abort.
FIXED 2026-10-11 (owner's pick, deploy.py): the board takes only signed writes, a participant only to its own
post/proof/quit slots (may_write); only a member's signed abort counts (check_abort), a failing participant just
quits; members drop invalid posts (bad signature, unloadable, duplicate) instead of failing (invalid_posts).
`BAD_POST=1 python deploy.py` covers the invalid posts. This script now expects both writes to be refused (403).
Run while a local demo is in its rounds (python deploy.py in another shell), from this folder:
    python gap_h.py abort|overwrite|equivocate|ahead|huge
It execs into participant container concilium-client-3 and does the write from there with its own credentials
(before the fix none were needed: the board didn't authenticate writers).
"""
import subprocess
import sys

mode = sys.argv[1]
code = {
    "abort": "import deploy as d; d.load('.concilium-demo/concilium.json', '.concilium-demo/p3.key', 'participant'); "
             "d.put(f'abort/{d.SESSION.hex()}', b'participant 3: I just felt like stopping')",
    "overwrite": "import deploy as d, time; d.load('.concilium-demo/concilium.json', '.concilium-demo/p3.key', 'participant'); "
                 "r = next(r for r in range(99, -1, -1) if d.try_get(f'roster/{r}', 'HEAD') is not None) + 1; "
                 "[d.put(f'post/{r}/4', b'junk' * 100) for _ in range(200) if not time.sleep(0.05)]",
    # from the review of the fix: equivocate on its own post, write far ahead, send a huge body
    "equivocate": "import deploy as d; d.load('.concilium-demo/concilium.json', '.concilium-demo/p3.key', 'participant'); "
                  "r = next(r for r in range(99, -1, -1) if d.try_get(f'post/{r}/3', 'HEAD') is not None); "
                  "d.put(f'post/{r}/3', b'different bytes' * 100)",
    "ahead": "import deploy as d; d.load('.concilium-demo/concilium.json', '.concilium-demo/p3.key', 'participant'); "
             "d.put('post/90/3', b'x' * 1000)",
    # one raw attempt (deploy.put retries a reset connection), then: did the board store it?
    "huge": "import deploy as d, sys, urllib.request as u\n"
            "d.load('.concilium-demo/concilium.json', '.concilium-demo/p3.key', 'participant')\n"
            "k, data = 'quit/' + d.SESSION.hex() + '/3', b'x' * (100 << 20)\n"
            "h = {'X-Party': d.PARTY, 'X-Sig': d.ME[0].sign(d.signed('PUT ' + k, data)).hex()}\n"
            "try:\n    u.urlopen(u.Request(d.CFG['board'] + '/' + k, data=data, method='PUT', headers=h), timeout=30)\n"
            "except Exception as e:\n    print(repr(e)[:120], file=sys.stderr)\n"
            "print('HTTP Error 413 (not stored)' if d.try_get(k, 'HEAD') is None else 'STORED', file=sys.stderr)",
}[mode]
r = subprocess.run(["docker", "exec", "-w", "/work", "concilium-client-3", "python3", "-c", code], capture_output=True, text=True)
refused = "HTTP Error 403" in r.stderr or "HTTP Error 413 (not stored)" in r.stderr
print(f"{mode}: {'refused by the board' if refused else 'NOT refused'} ({(r.stderr.strip().splitlines() or [''])[-1][:80]})")
assert refused, r.stderr[-500:]
