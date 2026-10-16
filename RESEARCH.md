# Privacy-preserving federated learning: where encrypted FL still leaks

Several parties (hospitals, banks, companies, phones: anyone with data they can't share) train one
shared model. Nobody's data leaves its owner. Each participant trains locally and sends only an
encrypted (CKKS) model update, and the server only ever sees ciphertexts. The data and the model are
a plug-in (see "Use it with your own data"); the medical datasets below are just the worked examples.
Each script adds one piece: it either attacks the previous design or closes the gap it leaves. Every
script checks its own claims with asserts, so a run that finishes without an error reproduces the table.

| script | what it shows | run |
|---|---|---|
| `fl_core.py` | encrypted FedAvg + encrypted poisoning filter matches plaintext (error ~1e-9) | `python fl_core.py` |
| `gap_b.py` | attacker defeats *encrypted* FLTrust by sending updates just outside the approximation's range | `python gap_b.py` |
| `gap_b_cnn.py` | same attack hides pneumonia on a chest X-ray CNN (non-IID) | `python gap_b_cnn.py` |
| `gap_c.py` | malicious server reads a single client's update via the key holder's decryptions; guarded key holder refuses | `python gap_c.py` |
| `gap_d.py` | one released aggregate is enough to recover the CKKS **secret key** → every client's update; noise added at decryption stops it, encrypted noise doesn't | `python gap_d.py` |
| `gap_e.py` | attackers who *know* the filter; the old length check collapsed the non-IID CNN, clipping to the median fixes it; exact FLTrust is no better | `python gap_e.py` (`cnn [seed]`: ~15 min) |
| `gap_f.py` | one oversized update breaks the committee's packed score decryption (round DoS) or drowns everyone else's scores in noise; per-participant fallback fixes it | Docker, below |
| `gap_g.py` | attackers who *optimize* against the filter (NDSS 2021 attacks + one tailored to its angle check): the filter holds with a secret reference, exact FLTrust collapses once, a known reference breaks it | `python gap_g.py` (LR, seconds); `python gap_g.py cnn <seed>` |
| `gap_h.py` | one participant could stop any run (unsigned abort, the board stored anyone's writes); fixed: signed writes checked by the board, member-only aborts, invalid posts dropped | live demo, see file |
| `post_crash_check.py` | fuzzed post bytes crash OpenFHE (SIGFPE); the members' loader process drops the post instead of dying | Docker, see file |
| `pine_fast_check.py` | the fast PINE wraparound checks give exactly the reference code's results | Docker, see file |
| `board_tls_check.py` | a client that never finishes the TLS handshake must not stall the board | `python board_tls_check.py` |
| `gap_i.py` | a participant reposts a transform of another's ciphertext; members could read the victim's coordinates; fixed: range proofs on by default with a tolerance from the rounding's real spread | Docker, see file |
| `eval_ss.py` | would secret sharing + PINE-style validity proofs keep the filter's protection? (plaintext, 5 seeds) | `python eval_ss.py` (~25 min, 16 cores) |
| `pine_bench.py` | PINE's reference code with 3 aggregators rejects the gap F wraparound | `python pine_bench.py` then Docker, see file |
| `range_proof.py` | participants prove \|u\| ≤ B with PINE on secret shares; members check the ciphertext holds the same `u` (random projections) before squaring it, so the flooding has a proven bound | Docker, see file |
| `flood_probe.py` | how big the committee's decryption noise must be: measures the error in what members decrypt (was 2⁵¹–2⁵⁵ against 2²⁰ of noise) and asserts `threshold.CT_ERROR` bounds it | Docker, see file |
| `profile_round.py` | where one threshold round's time goes, per phase and per OpenFHE call | Docker, see file |
| `threshold.py` | 3-member threshold keys (OpenFHE) + the `fl_core` range check computed on ciphertexts, so no single party can decrypt | Docker, below |
| `deploy.py` | the threshold protocol across machines (untrusted board server, committee members, participants) over HTTP(S), for any task file; locally as 14 containers | `python deploy.py` |

```
pip install tenseal numpy scikit-learn torch medmnist
python -c "from medmnist import PneumoniaMNIST; PneumoniaMNIST(split='train', download=True, root='.')"
docker build -t concilium-research .
docker run --rm -v "${PWD}:/work" concilium-research python3 threshold.py
```

## Use it with your own data

A task file tells `deploy.py` what the data and the model are. For tabular data in CSV files (yes/no
prediction) no coding is needed: copy `tasks/csv_task.py` next to the CSVs (`my_data.csv` per participant,
`root.csv` for the committee) and set the label column and scaling at its top (instructions inside; checked
2026-10-15: the breast cancer data as 12 CSV files, full deploy with attackers, 0.953, same as `tasks/lr.py`).
Otherwise copy `tasks/lr.py` (logistic regression) or `tasks/cnn.py` (a PyTorch CNN) and fill in its functions. To try it on one machine
(put it in this folder, since the containers only see this folder):

```
TASK=my_task.py SIMULATE=0 python deploy.py
```

For real, every party runs on its own machine. All it needs is Docker and the `concilium` image, which
carries the code (`docker build --target tool -t concilium .` once), plus a folder of its own for its keys,
the config, its data and its task file. `python friend/make_package.py release` makes `concilium-release.zip`:
the code, `build` and `concilium` scripts (`.sh` and `.cmd`) and `release/QUICKSTART.md`, a walk-through for all
four roles. Checked cold (2026-10-11): unzipped into an empty folder on Windows, 13 key pairs, the config and
all 14 parties started through `concilium.cmd`, the board reached through a published port like a separate
machine: finished, 0.953 (rounds 43 s median here: every message crosses Docker Desktop's port forwarder).
Every Python package is pinned by version and file hash (`requirements.txt`; pip refuses anything else)
and the base image by digest, so parties building months apart get the same packages. The tool image has no
TenSEAL (the research scripts' library; `docker build -t concilium-research .` is the research image with it). The
session id hashes the config **and the tool's code** (`concilium version` prints the code hash), so a party on
a different or modified build shows a different id and its signatures don't verify: the run stops at once.
On Linux with `openfhe` installed, `python deploy.py ...` works the same.

```
alias concilium='docker run --rm -v "$PWD:/work" concilium concilium'   # the board and members also need the network: see below
concilium keys alice                        # every party: alice.key stays on this machine, alice.pub goes to the organizer
concilium config concilium.json --board https://board.example:8000 --task my_task.py --committee m0.pub m1.pub m2.pub --participants p0.pub p1.pub ...
                                       # organizer: send concilium.json to everyone. Range proofs are on (--norm-bound 1):
                                       # pick the bound from your task's honest update norms, or --no-norm-bound
concilium board concilium.json cert.pem key.pem  # the server (HTTPS; plain HTTP without the files); publish its port: docker run -p 8000:8000 ...
concilium member concilium.json m0.key           # each committee member; the first listed leads
concilium participant concilium.json p0.key      # each participant; TASK=... points at its local task file
```

`--task lr` and `--task cnn` are the built-in examples; any other `.py` path is read from the party's
own folder. Checked cold (2026-10-02): 13 parties, each with an empty folder and only the image, keys
to finished run in 28 s for the LR task.

**Known limits, read before deploying.** Each is measured and explained further down.
- **Who you trust:** fewer than t committee members collude, and *every* member follows key generation honestly
  (no proofs of correct key shares yet). The board and the participants are not trusted.
- **What members see:** two scores per participant per round (plus 4 random projections for the range check),
  not the updates. The 1-bit MPC alternative was built and costs ~20× more (Next steps).
- **Noise is a mask, not differential privacy:** `--dp-z 0.1` gives ε ≈ 48 per round. A real ε costs 15–30 accuracy points.
- **Poisoning is bounded, not prevented:** worst seen 0.753 on the CNN under optimizing attackers. The reference
  (root) data must stay private to the committee, or attackers can hide a class completely.
- **Skewed (non-IID) data:** ~1/3 of honest updates fail the angle check and are left out of their round.
- **Failures:** any member can drop and rejoin, and so can the lead. A member dropping mid-round skips that round.
  The board can stop a run (it can't forge or read anything).
- **Size:** tested up to 107k parameters (2 GB RAM per member). Deploys ran with 10 participants; the protocol was checked with 20 (`threshold.check_groups`).
- **Tested on:** Windows + Docker Desktop, a Linux VM on the same laptop, and two Windows laptops on one Wi-Fi
  (2026-10-15), and GitHub runners (2026-10-16): Linux release-script deployment as non-root (found and fixed: the
  board's 5-deep accept queue overflowed behind Docker's Linux port proxy, which reset connections), Intel Mac via
  Colima (demo passed once, the second run hit the 90-min limit), ARM64 (`threshold.py`, OpenFHE built from
  source). Not yet across the internet.

`concilium.json` is who's who: everyone's public keys, the board URL, the task, `--min` (participants
needed for a round; default n − f − 1 with `--max-malicious f`, default 30%, so 6 of 10: the honest ones,
less one dropout, can always fill a round), `--round-timeout` (a round with fewer posts is skipped, and 3
skipped in a row stop the run; `WITHHOLD=1 python deploy.py` shows it), `--dp-z` (each member adds Gaussian noise of
std dp_z × norm bound to each round's sum, default 0.1; see the minimum-participants note under Next steps) and
`--deadline` (seconds to wait for stragglers). Every party must
hold the same file; each prints the session id (a hash of the file) at start, so compare it over a
second channel. The board is untrusted. Every message on it is signed by its author and bound to the
session and to its slot, so the board can delay or drop messages but can't forge, alter or move one.
Messages only the committee may read (roster, score partials, fallback votes) are also sealed under a
committee key that the lead hands each member via X25519. Without signatures, a malicious board could swap in its own joint public key
(participants would encrypt to it), replace a noise ciphertext or an aggregate partial (an update the
filter never saw), or give participants a fake model. `TAMPER=<board key> python deploy.py` makes the
demo board flip one byte of that message, and the run must abort (a tampered participant post is dropped
instead: that participant misses the round). The board also checks who writes (gap H): every upload is signed
by its sender, a participant may write only its own post, proof and quit slots, and nobody may overwrite a
participant's slot. That doesn't make the board trusted (a malicious board can still stop a run); it stops
a participant from doing what only the board could. If a member fails, it posts a signed abort and everyone
stops; a participant that fails only quits, and the lead stops the run once too few participants are left.
Members drop a participant's invalid post (bad signature, not a ciphertext, a copy of another post, bytes
that crash OpenFHE) instead of failing: `BAD_POST=1 python deploy.py`. `TLS=1 python deploy.py` runs the demo board over HTTPS with a throwaway
certificate (a private CA reaches the others through `SSL_CERT_FILE`).

| in the task file | what it is |
|---|---|
| `ROUNDS` | number of training rounds |
| `init()` | the initial model as one flat float64 vector (for PyTorch: `parameters_to_vector`) |
| `participant_data(i)` | participant i's own data; in a real deployment it reads that participant's local files |
| `root_data()` | a small clean labeled set held by the committee; the poisoning filter compares every update with the update trained on it |
| `train(w, X, y, seed)` | one local training run from model `w`, returning the update (new − old); deterministic for a given seed |
| `evaluate(w)` | optional: a one-line report printed at the end |

The protocol never sees inside the model, only update vectors, so any model that can be flattened
into a vector works. Cost grows with its size: one ciphertext per 8,192 parameters, about 1.3 MB of
upload per participant per round each. `SIMULATE=1` (the default) adds the demo's attackers, two
participant dropouts, a committee member crashing halfway, and a plaintext replay that checks every round, which needs everyone's data, so it's
only for testing. Assumptions: fewer than half the participants are malicious (the filter's median
needs that), the committee has the root set, fewer than t committee members collude (see below), and every member
follows key generation (signatures stop the board from swapping keys, not a member from choosing its
share badly; that needs proofs of correct key generation). Any committee size ≥ 2 and any number of
participants work: every 16 participants add one packed score group (2 more joint decryptions per
round; `threshold.check_groups` checks 20 participants with a 4-member committee).

**Member dropouts (t-of-n).** `config --threshold t` (default: all members but one, at least 2). After key
generation each member Shamir-shares its secret key share, split per RNS tower (OpenFHE's `ShareKeys` isn't in the
Python binding), and seals one share to each other member. A member that posts no noise within
`--member-timeout` s (default 60) is dropped for good. Each survivor then adds λ·(its share of the dropped key) to
its own key and decrypts as usual, so the partials still fuse and no one ever holds the dropped key
(`threshold.check_dropout`: the lead drops, 2 of 3 still score and aggregate, 1 alone can't decrypt).
The trade-off: any t members together can now decrypt everything. With t = n (the old behaviour), a single honest
member was enough to protect the updates; with t < n, fewer than t may collude. Participants accept a model signed
by t members. In the demo (3 members, t = 2), member 2 crashes in round 10: the run finishes all 20 rounds, every
round still matches plaintext (1.1e-11), and the dropout costs one 12 s round (the 10 s timeout). Any member can
drop at any point of a round, the lead included, and come back (2026-10-02). Every member posts a heartbeat
about every 0.5 s, and every wait on a member's message has one decider: the lead rules on the others, and the
lead's successor rules on the lead, finally. The decider posts a signed "ok" when the message arrives or "gone"
once its author's heartbeat stops for the member timeout, and every member, the author too, follows that one
ruling, so members can't split into different views. A ruling the lead should have posted is itself a lead
message, so a lead that dies while ruling is caught by its successor. On "gone" the round is abandoned (model
unchanged; a member dying after its aggregate partial is out leaves that round's update decryptable from the
board, which is what the model would have shown anyway), the member is dropped and its key shares folded in, and
the lowest member left leads. A dropped member rejoins by running the same command again: its key share and the
committee key are in `<key file>.state` (owner-only, written at keygen), everything else (context, evaluation
keys, the member list, the latest model if t members signed it) comes from the board. It posts a signed request,
the lead admits it in that round's member list, and every member checks that request's signature. Only members
ranked after the current lead rejoin (a former lead would take the lead back without the run's history).

| `CRASH=` | dropped (round: member) | result |
|---|---|---|
| `member` (round start) | {10: [2]} | 0.942, 84 s |
| `lead` (round start) | {10: [0]} | 0.942, 82 s |
| `member-mid` (after its score partials) | {10: [2]} | 0.942, 86 s |
| `lead-mid` | {10: [0]} | 0.942, 85 s |
| `member-rejoin` (restarted 30 s later) | {10: [2]}, rejoined {16: [2]} | 0.942, 88 s |

Rerun 2026-10-14 on the final code (need() decider chain, mask noise on; launcher waits 45 s before the rejoin):

| `CRASH=` | dropped (round: member) | result |
|---|---|---|
| `successor-mid` | {10: [1]} | 0.936, 138 s |
| `lead-mid` | {10: [0]} | 0.947, 134 s |
| `member-rejoin` | {10: [2]}, rejoined {15: [2]} | 0.947, 157 s |
| `lead+successor-mid` (`MEMBERS=4 THRESHOLD=2`) | {10: [1], 11: [0]} | 0.942, 166 s |
| `lead+successor` (`MEMBERS=4 THRESHOLD=2`) | {10: [1], 11: [0]} | 0.924, 152 s |

Every run matches plaintext round by round; a tampered ruling aborts (bad signature). The rulings cost ~0.3 s per
round (median 4.1 s, was 3.8). Restarted within the member timeout, a member just carries on without being
dropped. Left: with 3 members (2 to decrypt) the lead and its successor dying together leaves too few to
decrypt; the last member then stops the run with a signed abort, so participants quit at once (`CRASH=lead+successor`,
94 s; before 2026-10-15 it exited without the abort and participants waited 30 min for a model). To survive k members
dying at once, n >= t + k: with 4 members and threshold 2 the run survives it (table above), at the price of any 2
of 4 colluding being able to decrypt. A run that stopped goes on as a new run (new keys, new session) from the
latest model t members signed: `deploy.py export concilium.json model.bin`, then `config ... --init-model model.bin`
(the model goes into the config, so the session id covers it). `RESUME=1 python deploy.py` / `python demo.py
resume` does both (2026-10-15: abort, model after round 9 exported, resumed run 0.924 -> 0.936). A resumed,
converged model has honest scores u.ref of 1e-4..2e-3, near the decryption flooding (std 3.7e-4), so keep
decisions flip in up to 12 of 20 rounds vs plaintext; the self-check now accepts a flip only where a plaintext
score is within 8 flooding stds of 0; a dishonest decider can
rule a live member gone (it costs a member, gives no decryption power); the state file is plaintext at rest,
like the `.key` file.

**Demo:** `python demo.py` runs all nine steps in order with a one-line headline each (~20 min;
Docker must be running). `python demo.py gap_d threshold` runs a subset, e.g. for a short slot.

## Results (breast cancer data, 10 participants / 3 malicious, 20 rounds)

**Phase 1: encryption costs nothing in accuracy** (IID)

| setting | acc | honest rejected | malicious accepted | server s/round | KB/client |
|---|---|---|---|---|---|
| plaintext, no attack | 0.953 | 0 | 0 | – | – |
| encrypted, no attack | 0.953 | 0 | 0 | 0.012 | 324 |
| encrypted, attack, no defense | 0.105 | 0 | 60 | 0.013 | 323 |
| encrypted, attack, defense | 0.942 | 11 | 0 | 0.320 | 324 |

**Gap B: approximation attacks.**

Honest clients' squared update norms fall in 7.3e-4..0.218. Exact FLTrust with no attack: 0.947.
The attacker sends updates just outside the range the approximation was fitted on:

| deg | margin | newton | no attack | attacked | exact FLTrust, same attack | + 1-bit range gate | amplification |
|---|---|---|---|---|---|---|---|
| 7 | 2 | 0 | 0.942 | **0.105** | 0.936 | 0.942 | −6.1 |
| 15 | 2 | 0 | 0.953 | **0.374** | 0.936 | 0.942 | −66 |
| 31 | 2 | 0 | 0.947 | **0.117** | 0.936 | 0.942 | −72 |
| 7 | 2 | 2 | 0.942 | **0.117** | 0.936 | 0.942 | −6.1e5 |
| 15 | 2 | 2 | 0.947 | **0.117** | 0.936 | 0.942 | −1.6e15 |
| 15 | 10 | 2 | 0.942 | **0.117** | 0.936 | 0.942 | −1.9e16 |
| 15 | 10 | 3 | 0.947 | **0.374** | 0.936 | 0.942 | 3.5e48 |

A higher degree, a wider margin, or Newton refinement doesn't help: it only makes the blow-up
outside the range larger. Exact FLTrust is immune, and a 1-bit "norm² in range" gate brings
accuracy back to 0.942 in every configuration. For the attack points, real CKKS matches plaintext
to 1e-4..1e-3 (deg 7/15). At deg 31 and at newton 3 the CKKS value itself diverges (relative error ≥ 1).

**Chest X-ray CNN** (PneumoniaMNIST, 10 non-IID clients, 30 rounds). Exact FLTrust with no attack:
accuracy 0.816, pneumonia recall 0.972.

| approximation | no attack | attacked (3× max norm) | exact, same attack | + 1-bit gate |
|---|---|---|---|---|
| deg 7, no Newton | 0.848 / 0.949 | **0.625 / 1.000** | 0.838 / 0.962 | 0.833 / 0.951 |
| deg 15, 2 Newton | 0.832 / 0.954 | **0.375 / 0.000** | 0.838 / 0.962 | 0.845 / 0.949 |

(accuracy / pneumonia recall) With deg 15 the attack hides every pneumonia case. With deg 7 the
amplification is small and negative, so the model flips the other way (it flags everything as
pneumonia), but accuracy still collapses.

**Gap C: the key holder is a decryption oracle.** Against a naive key holder, all three attacks
recover client 0's update (one-hot reference: error 1.5e-6; swap aggregate and difference: 2.5e-9).
The guarded key holder takes only MAC-signed, fresh, full rounds, computes the result itself, and
releases only the new model (accuracy 0.942, the same as plaintext). It refuses replay, reused posts,
<8 clients, duplicated ciphertexts, and forged MACs.

**Gap D: key recovery.** The server re-encodes the decrypted aggregate, which gives it
`c0 + c1·s mod q` exactly, and solves a linear system for `s`:

| flooding std | aggregate error | key recovered | client 0 error |
|---|---|---|---|
| 0 | 1.0e-7 | **yes** | 5.1e-7 (leaked) |
| 0.1/scale ≈ 9e-11 | 1.0e-7 | **yes** | 5.1e-7 (leaked) |
| 1e-8 | 1.0e-7 | no | 7.5e9 |
| 1e-6 | 2.5e-7 | no | 8.1e9 |
| 1e-4 | 2.2e-5 | no | 8.2e9 |

Noise below 1/scale rounds away and doesn't help. Noise well above 1/scale (here 1e-8 ≈ 11/scale)
stops the linear attack at no accuracy cost. **Noise encrypted into the ciphertext doesn't help at
all** (key still recovered): the attack re-encodes whatever was decrypted and never needs the
message. This contradicted the original `threshold.py`; see below. Caveats: this uses textbook CKKS in numpy (N=64) because TenSEAL and OpenFHE
don't expose ciphertext polynomials, and it assumes the decryptor releases all complex slots.
Flooding is a heuristic here, not the formal version; `threshold.py` uses the formal sizing (below).

**Gap E: attackers who know the filter.** The filter (`fl_core.keep`) keeps an update if it
points the same way as the server's reference (`u·ref > 0`) and clips it to the median update
length. The 3 attackers collude. They estimate `ref` from their own honest updates and send updates
that stay just inside: angle θ toward a harmful push, length k× the median of their own honest
lengths. The push is either a label-flip update or a steady push on the output bias that hides the
disease. For comparison: ALIE and fl_core's sign-flip ×10. As in the protocol, a round with < 2
kept updates is skipped.

The first filter rejected updates longer than 3|ref| instead of clipping. On the non-IID CNN that
broke down: `ref` is sometimes short, so honest updates failed the length check (in one round they
were 4–17× |ref|). The kept set tilted toward one class, the model swung between "all normal" and
"all pneumonia", and the filter then rejected the other class's hospitals. Plain FedAvg without any
filter is stable (0.83–0.87 on 8 seeds), so the filter itself caused the swings. Clipping to the
median fixes it. It needs nothing new: `u·u` is already decrypted, and the scale is a plaintext
factor on the ciphertext, so there's no polynomial for gap B. With fewer than half the clients
malicious the median lies between two honest lengths, so the attackers can't raise it, and they
carry at most 3 median-lengths of the step.

Chest X-ray CNN, non-IID, worst case over all 7 attack settings, accuracy / pneumonia recall
(`python gap_e.py cnn <seed>`):

| seed | no defense | old filter (reject > 3\|ref\|) | filter (clip to median) | exact FLTrust |
|---|---|---|---|---|
| 0 | 0.375 / 0.000 | 0.832 / 0.862 | 0.829 / 0.838 | 0.651 / 0.844 |
| 1 | 0.000 / 0.000 | **0.375 / 0.000** | 0.808 / 0.913 | 0.625 / 0.944 |
| 2 | 0.000 / 0.000 | 0.689 / 0.664 | 0.686 / 0.790 | 0.630 / 0.874 |
| 3 | 0.000 / 0.000 | 0.625 / 0.790 | 0.769 / 0.905 | 0.753 / 0.933 |
| 4 | 0.000 / 0.000 | 0.694 / 0.826 | 0.812 / 0.926 | 0.655 / 0.805 |

Without attack, over the same 5 seeds: the old filter rejects 40–77% of honest updates and skips
0–15 of 30 rounds. The new one rejects 31–44% and skips none, at accuracy 0.835–0.872, which is
plain FedAvg's level. Under attack it skipped 6 rounds in total over 30 runs.

Breast cancer LR, worst case: IID 0.942 / 0.875 (old filter 0.936 / 0.859), non-IID 0.936 / 0.859
(unchanged). Exact FLTrust (0.942 / 0.906 and 0.942 / 0.875) would be free in `threshold.py`,
since the committee already decrypts both scores. On the CNN it's worse than the filter, probably
because honest non-IID clients get low trust. So the design keeps the filter.

Also tried, and dropped: a class-balanced root set (worse: 70–79% honest rejected), a reference
with momentum from past aggregates or a smaller server step (sometimes helps, still collapses on
some seeds), a reference summing the unit per-class root updates (more honest updates point the
right way, 76–81% instead of 62–71%, but it still collapsed on 2 of 8 seeds), and rejecting above
3× the median (one collapse to 0.375) or 2× (two, to 0.62 and 0.67): 3 attackers at up to 3
median-lengths can outweigh the honest mass, which clipping prevents.

What's left: about a third of honest updates still fail the angle check. Hospitals that see almost
only one class really do point away from the reference whenever the model leans toward their class.
For a single linear score, any update the angle check rejects would pass as a sign-flip of itself,
so no threshold on `u·ref` alone separates them. Seed 2's sign-flip still gets to 0.686 (the model
tips to "all pneumonia"). Caveats: 5 seeds, and no optimization-based attacks yet.

*Repeated draws (2026-10-10, `python gap_e.py reps <seed> 5`, GPU).* Each cell above is one draw of the
training randomness. Re-run with 5 independent draws per seed and attack (350 runs; no defense vs filter):

| | runs | accuracy < 0.7 | worst | median |
|---|---|---|---|---|
| filter, attacked (6 attacks) | 150 | 8 | 0.625 | 0.85 |
| filter, no attack | 25 | 1 | 0.657 | 0.84 |
| no defense, attacked | 150 | 30 (all 25 sign-flips at 0.000, 5 inside-flips down to 0.154) | 0.000 | 0.85 |
| no defense, no attack | 25 | 0 | 0.798 | 0.86 |

The filter never fell below 0.5 here, but one earlier draw (seed 0, inside-flip, a run of the per-class
experiment below; `results/2026-10-10/two_score_seed0.log`) did reach 0.375, so collapses are rare, not
impossible. 5 of the 8 dips below 0.7 are one draw (seed 2, draw 2) under 5 different attacks, and the
filter also dipped once with no attack (seed 0, draw 2: 0.657), while plain FedAvg never did (0 of 25).
That suggests the filter's own class swing on skewed data (honest rejection 30–36%) causes part of the
dips, but it is not shown: 1 of 25 vs 0 of 25 is not significant, and these runs didn't log how many
poisoned updates were kept (`gap_e.py reps` does now). The filter is what stops the sign-flip (no
defense: 25 of 25 at 0.000) and the inside-flip; the ALIE and hide attacks barely hurt FedAvg in this
setting. These runs used the GPU: same seeds as the CPU tables, but not the same draws (CUDA shuffles
differently and cuDNN isn't bit-reproducible), so compare distributions, not cells.

*Damping the swing (2026-10-10/12, pre-registered, `results/damp_stage1b_prereg.md`).* Server momentum 0.5
(V1: v = 0.5 v + agg, w += 0.5 v) against the plain filter (V0), CNN, no attack, 5 seeds × 200 fresh draws =
1000 pairs (`VARIANTS=V0,V1 gap_e.py damp <seed> 45 200 0`, `python results/damp_stage1b_analysis.py`):

| | runs < 0.7 | worst | 5th pct | median | mean | honest rejected |
|---|---|---|---|---|---|---|
| V0 filter | 8 | 0.655 | 0.764 | 0.846 | 0.835 | 33.1% |
| V1 momentum 0.5 | 4 | 0.681 | 0.784 | 0.853 | 0.842 | 34.1% |

Dips only one variant had: 7 vs 3, exact McNemar p = 0.34, so it fails the pre-registered bar (p < 0.05)
and damping is dropped. Stage 1 (draws 5–44, plus a smoothed reference that was worse) also failed. Momentum
leans better on every statistic, but the unattacked dip it targets is rare: 0.8% of runs (8 of 1000).

*A second score from per-class references doesn't help (2026-10-10).* Updates trained on the root set's
normal-only and pneumonia-only samples point almost opposite ways (cosine about −0.9): one class axis,
along which `ref` swings round to round. Hospitals with nearly one class are rejected about every other
round (47–53%), mixed ones 10–17%. Keeping by the direction orthogonal to that axis (`u·(e₀+e₁) > 0`,
e = unit per-class update) passes 83% of honest updates but also 17% of their sign-flips; with or without
reweighting the kept updates along the class axis toward the root's side, sign-flip ×10 collapsed it on
4 of 5 seeds (0.375–0.625), where the filter held, and honest rejection didn't drop (25–49%).

**Gap G: attackers who optimize against the filter** (`gap_g.py`, 2026-10-02). Gap E's attacks are fixed
recipes from a grid. These search for the most harmful update the filter still keeps, and adapt every round.
The attackers know the filter's rule but not the committee's secret reference:
- *AGR-tailored* (Shejwalkar & Houmansadr, *Manipulating the Byzantine*, NDSS 2021):
  their own honest mean plus γ × a perturbation (inverse unit vector, sign or std), γ as large as their
  simulation of the filter keeps (binary search).
- *Min-Max, Min-Sum* (same paper): as far as possible while staying no farther from their honest updates
  than those are from each other. Needs no knowledge of the filter.
- *Feedback* (tailored to our filter's one boundary, the angle check): tilt toward a harmful direction, read
  from the published model whether they were kept, tilt further if so and back off if not. Updates are sent
  10× long: the clip shortens them anyway, and 3 long updates lift the median the clip uses. Its
  "was I kept?" reading is crude, and the flip push is masked by honest drift.
- *White-box*: a bound, not an attack on our design. It knows the secret reference and sits at 89.5° from it.

Breast cancer LR: the filter's worst accuracy is 0.924 over all attacks, white-box included, while the
white-box flip takes undefended FedAvg to 0.076 (IID) / 0.591 (non-IID). Chest X-ray CNN, non-IID, seeds 0–4
(`python gap_g.py cnn <seed>`), worst accuracy / pneumonia recall:

| attacks | no defense | filter | exact FLTrust |
|---|---|---|---|
| none | 0.827 / 0.990 | 0.796 / 0.987 | 0.790 / 0.990 |
| 8 realistic (reference secret) | 0.375 / 0.000 | **0.753** / 0.997 | 0.375 / 0.000 |
| runs below 0.7 accuracy | 13 of 35 | **0 of 35** | 3 of 35 |
| white-box (reference known) | 0.000 / 0.000 | 0.527 / 0.254 | 0.732 / 0.997 |

The filter's claims hold against optimizing attackers: with the reference secret, its worst run is 0.753
against 0.796–0.880 without attack, never below gap E's worst (0.686), and pneumonia detection never collapses.
Exact FLTrust is weaker here: it collapsed once (seed 2, agr-tailored std: everything called normal). With the
reference known, the filter falls to 0.527 and misses 75% of pneumonia (seed 3, white-box hide): the reference
must stay secret, as `eval_ss.py` found. FLTrust's better white-box row isn't a fair comparison, since the
attack was aimed at our filter's boundary, where FLTrust gives it almost no weight. Honest rejection stays at
25–47%, the known skewed-data cost (gap E). Not tested: attacks that optimize a training objective end to end
(e.g. through a model of the whole aggregation), or more attackers than 3 of 10.

**Gap H: one participant could stop any run** (`gap_h.py`, found and fixed 2026-10-11). The board stored any
write, and the abort message was unsigned, on the reasoning that "the board can stop a run anyway". But any
participant can write to the board too. Verified on the LR demo: participant 3 uploading an abort stopped all
13 parties 18 s in; participant 3 writing junk over participant 4's post made the members fail its signature
check and abort. Fixed in three parts: the board accepts only signed writes and a participant only to its own
slots (`may_write`); only a member's signed abort counts, and a failing participant only quits
(`check_abort`, `run_role`, the lead's quit count in `close_round`); members drop invalid posts instead of
failing (`invalid_posts`). An independent review of that fix found five more holes, all closed: a participant
could overwrite its own post after the roster, so members read different bytes (participant slots are now
write-once, identical re-sends allowed); dropping both copies of a duplicated post let a participant censor
another every round (with range proofs a copy now fails only the copier's proof; without them both are
still dropped); unbounded uploads (64 MB cap per participant write, checked before reading, and only the
current or next round); a ciphertext at a lowered level would load but break scoring (posts must match a
fresh encryption's sizes, level and scale degree); old quits counted forever (only this round's missing
participants count). Verified on LR: `gap_h.py` abort, overwrite, equivocate (own post), ahead (round 90)
and huge (100 MB) are all refused during a live run, which finishes at 0.942; `BAD_POST=1` drops the junk
post and rejects only the copier's proof (without range proofs, both copies are dropped); `TAMPER=post/0/4`
drops participant 4 in round 0; `TAMPER=model/1` and `noise/0/1` abort; lead-mid crash, rejoin and TLS
pass. Heartbeat replay, fixed 2026-10-11: heartbeats were unsigned counts and any change counted, so replaying a
member's old ones (on plain HTTP, or by the board) kept a crashed member "alive" and the run waited forever. They
are now signed ns timestamps and only a higher one counts (a restarted member's keep rising). `REPLAY_BEAT=1
CRASH=member-mid python deploy.py` (the board answers half of heartbeat reads with a random old one): the crashed
member is dropped at round 10 and the run finishes (140 s); with the old reader the same run was still waiting
at 420 s. A replay can still win the first read of a wait, i.e. at most one member timeout.

Two more found 2026-10-11. (1) Hostile bytes that crash OpenFHE rather than raise: a fuzz of fresh
ciphertexts with a few bytes changed (same size, so the shape checks pass) killed or hung the deserializer in
6 of 60 (SIGFPE, an out-of-memory kill, a hang); in a member that stops the run. Members now load every post
in a spawned loader process first (`deploy.Loader`): if it dies or hangs, each post is retried alone and the
one that kills it is dropped. `post_crash_check.py` (SIGFPE from 8 header bytes); `BAD_POST=1` posts it in
round 6 and the run goes on (the round with the crash takes ~40 s instead of ~6 s: the loader restarts).
(2) The TLS board did every handshake in its one serving thread with no timeout, so a client that connected
and stalled stalled the board for everyone: `TLS=1 TASK=cnn` hung ~2 h at the round where the demo kills a
member, until every participant timed out and quit. Handshakes now run in each request's thread under a
60 s socket timeout (`board_tls_check.py`); the same run then finishes (0.845, 608 s). This doesn't explain
the one 970 s round seen 2026-10-10 (`CRASH=lead`): that run was plain HTTP. Not reproduced since: 13 more `NORM_BOUND=1 CRASH=lead` runs on current code (2026-10-11: 10 in a row, rounds ≤ 10.6 s, the crashed lead's container gone ≤ 0.4 s after its crash, so its heartbeat can't have kept it "alive"). Members now log the seconds per phase of any round over a minute, and the report lists them, so a recurrence names its phase. (3) A regression from
the gap H fix: the board let participants write one round past the latest roster, but a round whose lead
crashes has no roster, so after `CRASH=lead` every participant was refused the next round and the run stopped.
A published `model/r` now also opens round r; `CRASH=lead`, `lead-mid`, `member-mid` pass, and `gap_h.py ahead`
(round 90) is still refused. The board now logs each refused write, and a failed demo prints the board's log.

**Gap I: transformed copies of another participant's ciphertext** (`gap_i.py`, found and fixed 2026-10-11).
Members refused only byte-identical duplicates. Ciphertexts are malleable and participants hold the
evaluation keys, so a participant can post the victim's ciphertext plus an encryption of zero (a re-randomized
copy: the victim counts twice) or times a one-hot plaintext. On LR the masked copy lets the members decrypt
`u_k²` and `u_k·ref_k`, i.e. the victim's coordinate k (read −0.1049 for a true −0.1097, flooding included),
which matters when a member colludes with a participant. The CNN's scoring uses every multiplicative level, so
there a mask breaks scoring instead (reasoned, not run). Range proofs stop it: the attacker can only prove a
vector it knows, and the members' random-projection check compares that with the ciphertext. On LR every
transform was rejected (gap 0.07–0.27 against a tolerance of 0.004). At CNN size the old tolerance, 8σ of the
flooding plus the worst-case rounding `dim·2⁻¹⁵` = 0.43, let copies through (1 of 5 at |u| = 0.5, 5 of 5 at
|u| = 0.1). Rounding to the nearest step is uniform per coordinate, so its real spread on a ±1 projection is
`sqrt(dim/12)·2⁻¹⁵`, and the projections are drawn after every post is in, so a cheater can't line its rounding up
with them. With `8·sqrt(dim/12)·2⁻¹⁵` instead (0.011 at CNN size) and 4 projections instead of 2, 0 of 10
copies pass and 10 of 10 honest posts do. Range proofs are now on by default (LR rounds 4.9 → 6.7 s, CNN
~5 → 17 s); without them, gap I stands.

**Gap F: one oversized update vs. the packed scores** (`gap_f.py`, OpenFHE, in Docker). All
participants' `u·ref` and `u·u` share one ciphertext, so one joint decryption gives every score.
CKKS error grows with the largest value in a ciphertext, so one participant sending a huge update
hurts everyone else's scores:

| participant 0 sends | its \|u\|² | others' \|u\|² error, packed (true ≈ 0.2) | with fallback: decisions = plaintext |
|---|---|---|---|
| honest ×1 | 0.19 | 5.6e-12 | yes (no fallback needed) |
| honest ×10⁴ | 1.9e7 | 1.4e-8 | yes |
| honest ×10⁶ | 1.9e11 | 1.9e-4 | yes |
| honest ×10⁷ | 1.9e13 | 1.5e-2 | yes |
| honest ×10⁸ | 1.9e15 | **0.95** | yes |
| honest ×10⁹ and up | ≥ 1.9e17 | **decode fails** | yes |

Up to about 10¹⁵ the decryption succeeds, but everyone else's scores are silently noisy (at 10⁸×
they're pure noise, so honest decisions become random). Beyond that, OpenFHE's decode check throws
and the round can't finish: one participant stops training for everyone, every round, and nobody
can tell who. It never wrapped around to a plausible small norm, because OpenFHE's decode check
caught every case. An exact scheme (BFV, or MPC shares mod p) has no such check, so there the same
update could wrap silently. That's why a true 1-bit reveal needs input validity proofs (next steps).

Fix: if the packed scores don't decode, or hold a norm above 10⁶ × the median, members fall back
to decrypting each participant's two scores on their own ciphertext. The huge update then breaks
only its own score and is rejected. Members must all score the same way, so each posts a sealed
vote and any "fall back" vote wins. It reveals the same 2 scores per participant, and it costs
extra only when triggered. In the 14-container run with the 3 poisoners sending 10¹² × their
update (`NORM_BOUND=0 POISON_SCALE=1e12 python deploy.py`; with range proofs, now on by default, the proofs reject such
updates before scoring): all 20 rounds fall back and finish, accuracy 0.942,
every round matches plaintext. Rounds take 6.5 s instead of 3.9 s, plus 88 MB/round of board
traffic. A related fix: `keep` now drops an update whose clip factor rounds to 0, which before was
"kept" with no weight but still counted in the average's denominator.

**Threshold keys + range check (OpenFHE, 3 members, 3 poisoned clients)**

Each member computes `u·ref` and `u·u` for every client on the ciphertexts. `ref` comes from the
member's own copy of the model, so the server can't pick it. The committee jointly decrypts those
two scalars and every member applies `fl_core.keep` itself: it multiplies each kept client's
ciphertexts by its clip factor (a plaintext number) before summing. The comparison runs on decrypted
numbers, so there's no polynomial for gap B to attack. The cost is 2 revealed scalars per client,
not 1 bit.

| DP noise std | accuracy | max \|w − plaintext defended\| | honest rejected |
|---|---|---|---|
| 0 | 0.942 | 5.1e-3 (2.2e-11 before the flooding fix) | 13 (11) |
| 1e-3 | 0.942 | 1.5e-2 | 11 |
| 1e-1 | 0.953 | 1.5 | 20 |

(Undefended FedAvg under the same attack: 0.105.) With 2 of 3 shares, decoding fails. Replays and
an aggregate request without the scoring step are refused.

Gap D here: the members' encrypted noise is only a DP mask on the aggregate. What stops key
recovery is noise added inside each partial decryption. **Until 2026-10-02 that noise was far too
small** (`flood_probe.py`). OpenFHE adds a fixed N(0, 2²⁰) to each CKKS share (its formal
`NOISE_FLOODING_MULTIPARTY` mode is BGV/BFV only; `SetMultipartyMode` throws for CKKS). With the
default lazy rescaling, members decrypted ciphertexts still at scale 2¹⁰⁰, whose own error is 2⁵¹–2⁵⁵
in coefficients, so the share noise was ~34 bits *below* the error it was meant to hide. The ~1e-12
that repeat decryptions differed by was mostly OpenFHE's decoder adding noise after fusion, which
anyone fusing raw partials (the aggregate partials are on the board) skips.

Now each member rescales to 2 RNS towers first (scale 2⁵⁰, measured error 2^5.4–2^5.9) and adds its
own flooding with σ = √(12τ)·2^(s/2)·ct.t, OpenFHE's rule for single-key CKKS (`NOISE_FLOODING_DECRYPT`):
s = 30 bits of statistical security, τ = 2¹⁶ decryptions per key, ct.t ≤ 2⁷, so σ ≈ 2^31.8. Every member
adds the full σ, so one honest member suffices. The noise is a complex-Gaussian plaintext added to the
share; that needs the COMPLEX data type, because REAL plaintexts silently drop imaginary parts and would
flood only half the coefficients. `fuse` therefore checks the imaginary parts itself (gap F's "decode
fails"). The run checks that the measured noise is the predicted 3.7e-4 per value, which is the price:
0.942 accuracy as before, the model 5e-3 from plaintext after 20 rounds (was 2e-11), and a borderline
keep/reject decision can flip. Partials got 3–4× smaller (board traffic for the LR deployment: 81 → 62 MB per round).

**Bounding `u` before squaring it (`range_proof.py`; in `deploy.py` with `--norm-bound`).** The flood
above is sized from measured errors, the average-case sizing [Guo et al.](https://usenix.org/conference/usenixsecurity24/presentation/guo)
warn about. `u·u` is the one ciphertext × ciphertext product, and past |u| ≈ 10 its error grows with
|u|² (`flood_probe.py`: 2^9.7 at |u| = 100, 2^13.4 at |u| ≈ 356), so a participant sending a huge `u`
could outgrow any fixed σ. Only members see fused scores, so that takes a corrupt member plus a corrupt
participant. `u·ref` and the aggregate (products with plaintexts) don't grow with |u|. The fix bounds `u`
first:

1. Each participant also secret-shares `u` to the members with a PINE proof that |u| ≤ B (the vendored
   reference code; it now runs in the `concilium-research` image through a 100-line stand-in for the bits of SageMath it
   used, `pine_poc/nosage`).
2. A proof about shares says nothing about the ciphertext, so after all posts are in, the members draw 2
   random ±1 vectors r and compare `u·r` from the ciphertext (a plaintext product, decrypted with flooding)
   to `u·r` from the shares. A ciphertext off by d passes a check with probability about tol/|d|.
3. Only participants that pass are scored, and the flooding is sized for the proven bound:
   `ct_error(B) = 2⁷·max(1, B²/2⁹)`, fitted with ≥ 1 bit of margin to every row `flood_probe.py` measures
   (B ≤ 22 costs nothing extra; B = 100 makes the noise ~20× larger).

| participant (10, B = 1, honest \|u\| ≤ 0.47) | PINE | max \|u·r(ciphertext) − u·r(shares)\| (tol 3.9e-3) | scored |
|---|---|---|---|
| 7 honest | ok | 5e-5 – 8e-4 | yes |
| huge ciphertext (10⁶·u), valid small shares | ok | 2.6e5 – 5.7e5 | no |
| ciphertext off by \|d\| = 0.5 | ok | 0.38 – 0.71 | no |
| \|u\| = 100·B, lying norm proof | rejected | – | no |

The 7 that pass are scored as usual; their |u|² decrypts within 5e-4–1.1e-3 (2 runs; the B-sized flooding). Costs: members see 2 random
projections of each update besides the 2 scores, a share plus proof is ~30 KB per member, and the pure-Python
reference code took ~12 s per participant to verify (a production build would use a compiled PINE). Sped up 2026-10-11 without leaving Python: 90% of PINE's time was the wraparound checks, one field object per coordinate; done as integer sums with numpy-decoded draws (bit-identical, `pine_fast_check.py`) PINE at CNN size goes from 17.6 to 2.2 s per proof + 3 verifications, and the CNN deploy's median round from 18.6 to 9.9 s. On 3
aggregators: the PINE paper is written for 2 servers, but the IETF PINE draft defers its security to Prio3's,
which allows any number of aggregators with one honest for privacy (VDAF draft 9.8). The same spec (9.7) requires
Field128 with >= 1 proof, or Field64 with >= 3, for proofs that use joint randomness (a client can otherwise grind
offline for a false proof); the reference `Pine64` has Field64 with 2. Since 2026-10-11 this uses `Pine128`
(1 proof): same cost at CNN size (prove 2.0 vs 1.8 s, verify 1.6 vs 1.5 s per member, proof 0.27 vs 0.14 MB;
`results/2026-10-11/pine_variants.py`). B is public and must admit honest updates; the fl_core
attackers' −10× updates (|u| ≈ 4.4) no longer fit under B = 1, so PINE alone rejects them.

In the deployment (on by default since 2026-10-11 with B = 1; `NORM_BOUND=0 python deploy.py` or config
`--no-norm-bound` turns it off) participants post their PINE shares
sealed per member (a one-time X25519 key, the whole post signed) before their ciphertext, and the lead counts a
participant only once both are up. Members commit to their share of the projection randomness before anyone
reveals, so no member can steer r. PINE shares are n-of-n, so the lead publishes each round's member list and a
round in which it changes is skipped (fails safe), as is one with fewer than `min_participants` proven. Honest
participants clip to B. In the demo participant 0 sends its −10× update unclipped with a lying proof:

| LR, B = 1, 20 rounds | result |
|---|---|
| deployed model | accuracy 0.942 (same as without proofs) |
| rejected by the range proof | participant 0 in rounds 0–4, exactly the rounds its update was over B; later training converged and its −10× update fit under B, so its proof was true and the filter handled it |
| honest participants rejected by the range proof | 0 |
| rounds not scored | 1 (round 10: member 2 crashed after participants had split their shares) |
| each scored round vs plaintext | max error 3.3e-4, 1 borderline decision differing |
| round time, median | 5.6 s (3.7 s without proofs; 45 s with PINE's arithmetic as vendored, 2026-10-02). Per participant at d = 31: proving 0.2 s, verifying 0.2 s per member, was 16.5 s and 6.1 s. The reference code multiplied and interpolated polynomials in O(n²) over wrapped field objects; `pine_poc/vdaf_poc/field.py` now keeps plain ints, multiplies by Kronecker substitution (one big-int product) and interpolates with an NTT (the FLP's points are roots of unity). Same outputs (`python3 -m vdaf_poc.field` checks them against the reference; bench.py and range_proof.py still reject both cheats). At the CNN's 14,146 values: proving 4.6 s, verifying 3.8 s. Members verify in parallel worker processes (spawned: OpenMP isn't fork-safe). Whole run 122 s (was 14 min); the demo's 30 s deadline is back to the default 3 s |
| extra board traffic per round | 1.1 MB proofs + 4.6 MB PINE prep and projection partials |

**CNN with range proofs** (`TASK=cnn NORM_BOUND=1 python deploy.py`, 2026-10-10). Honest CNN updates have
norms 0.1–1.1, so B = 1 clips almost nothing. Result: accuracy 0.848, pneumonia recall 0.933 (plain CNN runs:
0.78–0.86); the liar is rejected in every round (its 10× update is never under B); 1 round not scored (the
member crash); 53 honest updates rejected by the angle check (as without proofs); rounds 17 s median, whole
run 9 min. The first try skipped 21 of 30 rounds (accuracy 0.375): the lead's 3 s straggler deadline ran from
the round's start, not from when 8 participants had posted as documented. Proving takes > 3 s on the CNN, so
the lead closed at exactly 8, one of them the liar, leaving 7 proven < 8. Fixed in `close_round`; `deploy.py`
now asserts that at most a tenth of rounds are skipped. Still true: when exactly the minimum post and one
proof fails, that round is skipped (the lead can't verify proofs alone).

Implementation note: fusing the same partials isn't bit-exact (results ~1e-14 apart). If each
member kept its own fused model, their `ref` encodings would differ, the score ciphertexts would
differ, and fusing their partials would give garbage. So the lead member publishes the model and
the others adopt it after checking it against their own fusion (within 1e-6). The same applies to
the clip factors: each member fuses the scores itself, and the results differ by ~1e-12, which is
enough to make members scale by different numbers and turn the fused aggregate into garbage (it did).
`keep` rounds the factor down to a multiple of 2⁻¹⁰, so members disagree only within 1e-12 of a step,
the same odds as a keep/reject tie.

Speed: the scores are packed. Each member rotates client j's ciphertext into slot block j, so one
multiply gives every `u·ref` and one gives every `u·u`. It then block-sums and masks everything
except slot 32·j before decrypting; unmasked, the partial sums in the other slots would reveal
individual coordinates. That's 2 joint decryptions per round instead of 20. Rotations (`EvalSum`
at 93 ms) were 87% of the runtime, and the full run went from ~8 min to 2.4 min. The later
general layout (chunks of 8,192 values, for the CNN) costs more rotations for small models, and
the in-process run is back to 5.7 min. The same protocol in `deploy.py`, with members in
parallel, takes 65 s.

**Deployment (`deploy.py`)**: 1 untrusted board (the server), 3 committee members and 10
participants, each a separate container talking over HTTP. Each member generates its own key share
and posts only public keys, so secret shares never leave their container. Every party has its own
signing key and the demo launcher writes the shared config, exactly as separate machines would (see
"Use it with your own data"). The board is parsed with length-prefixed framing, never pickle.
`python deploy.py` runs breast cancer; `TASK=cnn python deploy.py` runs the chest X-ray CNN.

| | breast cancer LR | chest X-ray CNN (non-IID) |
|---|---|---|
| parameters → ciphertexts per update | 31 → 1 | 14,146 → 2 |
| rounds | 20 | 30 |
| deployed model | accuracy 0.942 | accuracy 0.78–0.86, mean 0.805 over 5 runs with the flooding; 0.79–0.82, mean 0.804, over 5 runs of the same code without it (2026-10-02): no measurable cost, more spread |
| each round vs one plaintext defended round from the same model | max error 3.7e-4 (flooding; was 9.5e-12), decisions differing in 1–2 of 20 rounds | 5.7e-4–8.3e-4 (was 2.2e-11), 0–1 of 30 |
| rounds skipped (< 2 updates kept) | 0 | 0 of 30 (9 with the old filter) |
| round time, median | 3.8 s (0.9 s when everyone posts on time) | 4.7 s (was 5.4 s before the deserialization fix, see Performance) |
| whole run | 65–70 s | 157 s (was 180 s); 178 s on a later run when the laptop was slower overall (plain training +33%) |
| upload per participant per round | 1.3 MB | 2.6 MB |
| board traffic per round (in + out) | 62 MB (81 before partials were rescaled) | 149 MB before the rescaling |
| key generation | 3.6 s; 436 MB posted, 792 MB fetched (28 rotation keys) | same |

- **Scores are sealed from the board:** fusing partial decryptions needs only the public context,
  so whoever holds all 3 partials can fuse them. Members used to post their score partials in the
  open, which handed every participant's `u·ref` and `u·u` to the untrusted board each round. They
  are now AES-GCM sealed under a committee key (the board slot name is bound in, so a blob can't
  be replayed into another round). The run checks that the board's copy doesn't parse. The
  aggregate partials stay open, since fusing them only gives the model update the lead publishes
  anyway. The members themselves still see both scores; see next steps.
- **Dropouts:** participant 9 is offline for rounds 5–9, and participant 8 is gone from round 12. The
  lead member closes each round 3 s after ≥ 8 participants have posted (6 since 2026-10-11, see Next steps),
  and posts the roster. All members must score exactly the same ciphertexts, or the joint
  decryption is garbage, so they all use that roster, after checking it (distinct ids in range,
  ≥ 8). A bad roster can't isolate one client, because `verify_posts` still demands ≥ 8.
- **Skipped rounds:** when the filter keeps fewer than 2 updates, members refuse to decrypt (a
  "sum" of one update is that update). The model stays unchanged for that round. With the old
  filter this happened in 9 of 30 CNN rounds. Clipping to the median (gap E) brought it to 0,
  with 72 honest updates rejected over the run (~38%, all on the angle check).
- **Why it's checked round by round:** CNN training amplifies differences. A 1e-13 CKKS
  difference can flip a borderline decision later, so a full 30-round plaintext replay drifts
  apart (0.09 in weights) even though every single round is exact. For the same reason, results
  vary between runs (accuracy 0.78–0.86, pneumonia recall 0.91–0.98 over 10 runs on 2026-10-02), because the fresh partial-decryption noise
  differs each time. Training is seeded per (round, participant) and single-threaded, so the lead
  can replay any participant's round.
- **Threads:** all containers share one host. With OpenMP's default, each member uses all 16
  cores and they oversubscribe (13.8 s rounds). With each member given cores/3, rounds take 4.0 s.
- **Layout:** an update is split into 8,192-value ciphertexts. Each client's per-chunk products
  are summed and folded down to a 512-slot block (4 rotations). Every block then holds the sums, so
  masking block j puts the client there. One block-sum gives the scores of 16 clients, so 2 joint
  decryptions per round for every 16 participants. This needs
  multiplicative depth 3, so ciphertexts are ~25% bigger than before.
- **Two machines (2026-10-02):** `REMOTE_MEMBER=1 python deploy.py` runs member 1 outside Docker,
  in a WSL Ubuntu VM (its own OS, Python and key file, reaching the board across WSL's NAT and
  Docker's published port). Same results as one host (accuracy 0.942, max error 1.2e-11, member 2
  dropped exactly at its crash), 80 s with HTTP and 88 s with TLS. The VM shares the laptop's CPU and
  RAM, so it is a separate machine on the network, not separate hardware. The first attempt took 1,760 s
  and dropped an honest member: Docker Desktop's port forwarder silently dropped some new
  connections (the lead sat in SYN_SENT for 19–36 s at a time), and requests had no timeout, so each
  drop cost a full TCP backoff. Every board request now times out after 5 s and is retried. A
  stalled GET counts as "not posted yet", a PUT is re-sent, since the board just overwrites the slot.
  In the demo, local containers also reach the board directly, not through the forwarder.
- **A friend's computer over the internet (Tailscale, free):** member 1 runs on another person's computer,
  everything else on this laptop. Their side is `concilium-friend.zip` (`python friend/make_package.py`): Docker
  Desktop, Tailscale, and three numbered scripts explained in `friend/README.md`. This side:
  1. Install Tailscale here and sign in. In the Tailscale admin console, share this laptop (Machines → this
     laptop → Share) with your friend's Tailscale account. Sharing lets their computer reach this laptop only,
     and this laptop never needs to reach theirs.
  2. Let the board through Windows Firewall from Tailscale addresses only (PowerShell as administrator):
     `New-NetFirewallRule -DisplayName "concilium board via Tailscale" -Direction Inbound -Protocol TCP -LocalPort 8000 -RemoteAddress 100.64.0.0/10 -Action Allow`
  3. Send them `concilium-friend.zip`. They build the image and send back `m1.pub` (never `m1.key`); put it in
     this folder.
  4. `REMOTE_MEMBER=1 BOARD_HOST=concilium-board python deploy.py`. When it says so, send them
     `.concilium-demo/concilium.json` and this laptop's Tailscale address (`tailscale ip -4`, starts with `100.`).
     They run `3_run`, which maps the name `concilium-board` to that address. The launcher waits 30 min for them.
     The member timeout is 120 s here (10 s locally), since a remote member fetches each round's ~15 MB
     through this laptop's upload.
- **A friend's computer on the same Wi-Fi (2026-10-15): PASS.** Member 1 on the friend's own Windows laptop
  (fresh keys, image built from `concilium-friend-windows.zip`), everything else on this laptop; no Tailscale: the
  friend's `3_run` gets this laptop's LAN IP, and the firewall rule is `-RemoteAddress LocalSubnet` instead.
  Accuracy 0.930, keep decisions differing from plaintext in 0 rounds, member 2 (local) crashed at round 10 as
  planned and the friend's member carried the decryption with member 0 after it. Keygen 276 s (436 MB posted,
  792 MB fetched), rounds 60 s median (12-361 s; the slowest is the crash round), 30 min in all.
- **Shortcut:** the example tasks load the full dataset and give each participant its shard; a real task reads only local files.

## Evaluation: secret sharing with PINE for a true 1-bit reveal

The committee members still see 2 scores per participant. A true 1-bit reveal needs exact arithmetic,
and exact arithmetic brings back the wraparound from gap F, so every input needs a validity proof.
Instead of range proofs over HE ciphertexts, this evaluates secret sharing: each participant splits its
update into 3 shares, one per member, plus a PINE proof. [PINE](https://arxiv.org/abs/2311.10237)
(Rothblum et al., USENIX Security 2024; [IETF draft](https://datatracker.ietf.org/doc/draft-chen-cfrg-vdaf-pine/))
lets the members check |u| ≤ B for a **public** bound B. It catches the wraparound, and the members
learn only valid/invalid plus the sum of the valid shares.

**PINE's reference code works for us** (`pine_bench.py`, SageMath, 3 aggregators): honest updates
aggregate (error 3e-5, 15 fractional bits). The gap F attack is rejected by the aggregators' wraparound
checks: a coordinate x = 1,099,494,850,304 with x² ≡ 2 (mod p), so the squared-norm check alone passes.
An update at 100× the bound that lies about its norm is rejected by the norm check. The proof's fixed
cost is about 3,000 field elements: +75% at d = 4,000 and about +25% at our CNN's 14,146, so ~140 KB per
upload instead of 2.6 MB of CKKS. The reference code took ~45 s per report in SageMath; our faster field arithmetic (deployment table
above) brings it to ~1-6 s.

**But PINE can't express our filter**: it has no median (a data-dependent bound) and no angle check
(`u·ref > 0` with the members' reference). Plaintext simulation on the CNN (`eval_ss.py`, gap E
attacks, 5 seeds, worst accuracy over the 6 attacks):

| rule | per seed | mean | attacked runs < 0.7 | no attack |
|---|---|---|---|---|
| today: clip to median + angle check | 0.816 0.785 0.652 0.654 0.798 | 0.741 | 2 | 0.840–0.869 |
| clip to median, no angle check | 0.375 0.607 0.383 0.771 0.827 | 0.593 | 3 | 0.782–0.862 |
| **PINE bound B = \|ref\| + angle check** | 0.843 0.659 0.641 0.647 0.832 | **0.724** | 3 | 0.817–0.859 |
| PINE bound B = \|ref\|, no angle check (PINE alone) | 0.375 0.625 0.388 0.625 0.625 | 0.528 | 5 | 0.806–0.867 |
| PINE bound B = 3\|ref\|, no angle check | 0.375 0.375 0.375 0.375 0.625 | 0.425 | 5 | 0.824–0.867 |

(Single cells move by up to ~0.12 between runs: the "today" row differs from the gap E table above only
in the thread count. Compare means.)

- A public per-round bound B = |ref| (participants clip themselves, PINE proves it) is about as good
  as the median. So the norm side fits PINE.
- The angle check is not optional: without it, a sign-flip clipped to the bound still collapses the model.
- Publishing ref, so participants could prove `u·ref ≥ 0` inside the same proof, fails: attackers who
  know ref exactly push their harm at 45–89.9° from it. 8 of 40 runs drop below 0.7, and seed 3 loses
  **all** pneumonia detection (0.375 / 0.000 at 70° and 85°). So ref stays secret to the members.

**Resulting design (not built):** participants secret-share their clipped update among the 3 members
with a PINE proof for |u| ≤ |ref| (B published each round). Each member computes its share of `u·ref`
locally, since it's linear in u, and the members run one secure comparison per participant for the
sign. Members learn 2 bits per participant (PINE valid, angle ok) and the aggregate. No HE: gap D (key
recovery) and gap F go away, and uploads shrink ~20×. Costs: the members need a comparison protocol for
at least 1 honest of 3, e.g. MP-SPDZ's dishonest-majority protocols. [SAFEFL](https://eprint.iacr.org/2023/555)
runs all of FLTrust in MP-SPDZ with 3 parties and reports ~16× the communication of plain FedAvg with
malicious security; here only the comparison runs in MPC, since PINE covers input validity. Open
questions: PINE's security proof assumes 2 servers (the reference code takes 3 shares); `u·ref` on
shares needs `ref` in the same fixed-point field; the project would stop being an HE project.

**Prototype (2026-10-13, `ss_mpc.py`, `Dockerfile.mpc` = the concilium-research image + MP-SPDZ 0.4.3's release binaries).**
PINE leaves each member an additive Field128 share of u; each member computes its share of `u·ref` locally and
the three feed them, unchanged, into MP-SPDZ run with PINE's prime, which reveals only `u·ref > 0`. Checked
(asserted): 5 FedAvg rounds of the LR task with 3 sign-flip attackers clipped to B = |ref|: the MPC bits equal
the sign of what the shares add up to and of the float `u·ref` in every round, every attacker is rejected, and
the model reaches 0.918. Cost of the comparison alone, 3 members on this laptop (no network latency):

| protocol (MP-SPDZ) | security | per participant, all members | time | 10-participant round |
|---|---|---|---|---|
| `mascot` | malicious, 2 of 3 may be corrupt | 44 MB | 56–69 ms | 440 MB |
| `mascot -Y` (edaBits) | same | 105 MB | 270 ms | 1 GB |
| `semi` | semi-honest, 2 of 3 | 7 MB | 10–13 ms | 70 MB |
| `semi -Y` | same | **0.37 MB** | 2 ms | 3.7 MB |
| `lowgear` (HE triples) | malicious, 2 of 3 | not measured: 7.5 min key generation, then failed under a 3 GB container cap (likely memory; not diagnosed) | | |

Per-participant cost is flat from 10 to 1000 participants (MASCOT: 44 GB for a 1000-participant round).
Compare the CKKS deployment's member traffic per 10-participant LR round: noise + scores + aggregates ≈ 19 MB
posted + fetched via the board (deploy report). So with malicious security, the one comparison costs **at
least ~20× the whole CKKS scoring step** in traffic: ~150 MB sent per member, ~12 s at 100 Mbit/s before
latency, against today's ~6 s rounds. Semi-honest with edaBits is
cheap, but a member that deviates isn't caught. The participant side does get cheaper (PINE shares, no 2.6 MB
ciphertext upload), and gaps D and F disappear with HE.

Found while building it: **a member can lie about its own share** and flip any participant's bit. MASCOT
authenticates what a member inputs, not that it is the share PINE gave it. In the CKKS version that can't
happen: every member computes on the same public ciphertexts. Fixing it needs shares the participant
authenticates (e.g. MACs under a key the members hold jointly, checked inside the MPC), which costs more MPC.
Verdict: the design works and reveals 1 bit, but on this evidence it doesn't beat the CKKS version for a
3-member committee with malicious security. Not tried: HighGear/LowGear with enough memory (their triples
are often cheaper than MASCOT's on fast links), a smaller field for the comparison.

## Performance: where a round's time goes

Setup (2026-09-26): Intel i7-13620H (10 cores = 6P + 4E, 16 threads, AVX2, **no AVX-512**), NVIDIA RTX 4060
Laptop GPU (8 GB, CUDA-capable driver). HE: OpenFHE 1.5.1 (the `openfhe` pip wheel, in Docker), built with
OpenMP (libgomp) and without Intel HEXL; HEXL's big wins need AVX-512, which this CPU doesn't have.
TenSEAL 0.3.18 is used only by the phase-1/gap scripts. PyTorch is a **CPU-only build** (`2.14.0+cpu`) on
the host and in the image, so the GPU is never used. `OMP_NUM_THREADS` is unset, so OpenMP uses all 16 threads.

One CNN round (`profile_round.py`: 14,146 parameters = 2 ciphertexts per participant, 10 participants,
3 members run one after another in one process, CKKS ring dimension 16,384, depth 3):

| phase | 16 threads (default) | 4 threads |
|---|---|---|
| scoring: `u·ref`, `u·u` on ciphertexts (3 members) | 4.53 s | 4.20 s |
| per-participant fallback (forced here; runs only under gap F attacks) | 3.50 s | 3.14 s |
| participants' training (PyTorch, CPU) | 1.29 s | 1.32 s |
| all threshold decryption: partial decrypts + fusion | 0.64 s | 0.61 s |
| aggregation: scale, sum (excluding its decryption) | 0.29 s | 0.28 s |
| participants' encryption (10 × 2 ciphertexts) | 0.18 s | 0.18 s |
| **total** | **10.5 s** | **9.9 s** |

By OpenFHE call: 888 rotations take 4.6 s (5.2 ms each), 984 additions 1.8 s, 270 multiplications 1.2 s.
Rotations and additions dominate; decryption is 6%. The fallback accounts for 60% of the rotations when it runs.

**Fixed (2026-09-26): deserialized ciphertexts were slow.** Every operation on a ciphertext loaded with
`DeserializeCiphertextString` (all of them: they arrive through the board), and on anything computed
from it, ran slower: an add took 2–3 ms instead of 0.12–0.16 ms on freshly encrypted ones, at the same level, scale
and slot values (measured in isolation; `Clone()`, `x * 1.0` and `x + 0.0` don't help). Adding it once to
a *local* ciphertext rebinds it: `Member` keeps `EvalSub(e, e)` of a local encryption (all-zero elements,
so the 3 members still compute bit-identical ciphertexts, which fusion needs) and adds every loaded chunk
to it (~1 ms per ciphertext). Also, after the fold every block holds the sums, so a block-j mask replaces
"mask block 0, rotate to block j" (54 fewer rotations per round). Same run, 4 threads:

| phase | before | after |
|---|---|---|
| scoring (3 members) | 4.20 s | 2.67 s |
| per-participant fallback (forced) | 3.14 s | 1.77 s |
| participants' training | 1.32 s | 1.32 s |
| all threshold decryption | 0.61 s | 0.68 s |
| aggregation | 0.28 s | 0.21 s |
| participants' encryption | 0.18 s | 0.18 s |
| **total** | **9.9 s** | **6.9 s** |

Additions now cost 0.24 ms, multiplications 2.8 ms (was 4.3), rotations 3.5 ms (was 4.5). The deployment
(`TASK=cnn python deploy.py`) went from 180 s to 157 s, with a 4.7 s median round (was 5.4 s). The
deployment is now bounded by the 3 s participant deadline (a simulated dropout hits 23 of 30 rounds) and
by training, not HE. Members run with `OMP_WAIT_POLICY=PASSIVE`: idle OpenMP threads sleep instead of
spinning, which takes 9% longer but uses 54% less CPU time (profile at 5 threads: 33.8 s → 15.7 s CPU) and
leaves cores for the participants' training on the same host.

Checked and not done:
- **Threads inside a member**: openfhe-python holds the GIL (8 threads: 0.7–0.9× of sequential), so it
  would take a process pool. libgomp isn't fork-safe, and with spawn each worker must load the keys and
  every ciphertext has to be serialized both ways. Not worth it while rounds are deadline-bound.
- **CKKS parameters** (single-key contexts, same 128-bit security): FLEXIBLEAUTO instead of FLEXIBLEAUTOEXT,
  2 key-switching digits, or BV key switching make rotations ~20–30% cheaper. They also change the
  ciphertext error that the flooding is sized from, so they'd need `flood_probe.py` re-run (it asserts the
  bound). Kept the defaults.
- **Hoisted rotations** don't fit: the fold and block sums rotate a different ciphertext each step.

- **OpenMP barely helps**: 1 thread takes 12.5 s, 16 threads 10.5 s, 4 threads 9.9 s. With ring dimension
  16,384 and at most 4 RNS limbs, there's little to parallelize inside one operation. On single
  operations, 16 threads are clearly oversubscribed: a rotation takes 31 ms at 16 threads, 4.5 ms at 4 and
  9.9 ms at 1 (medians of 40). So the 100% CPU is mostly OpenMP threads spinning. The parallelism that
  is really there is *across* participants and members (10 × 3 independent pieces of work). `deploy.py` already runs the 3
  members in parallel (with cores/3 threads each), but each member scores its participants one by one.
- An addition cost 1.8 ms in the round but 0.2 ms on two fresh ciphertexts: deserialization (fixed above),
  not rescaling.
- The long runs in this project (`gap_e.py cnn`, `eval_ss.py`: 15–40 min) are pure PyTorch training on the
  CPU-only build, not HE.

### Model size (2026-10-11)

Same deployment (3 members, 10 participants, 30 rounds, range proofs and noise on, all 14 parties as
containers on this laptop), the small CNN against `tasks/cnn_wide.py`: the same network with wider layers.

| | CNN | wide CNN |
|---|---|---|
| parameters / ciphertexts per update | 14,146 / 2 | 107,586 / 14 (7.6× / 7×) |
| accuracy, pneumonia recall | 0.742–0.859 (7 runs) | 0.796, 0.997 (1 run) |
| round, median | 9.9 s | 41.9 s (4.2×) |
| whole run | 342 s | 1,459 s |
| upload per participant, round | 2.6 MB | 18.6 MB (17.0 post + 1.65 proof) |
| through the board, round | ~130 MB | ~870 MB |
| peak memory: member / participant / board | 1.1 / 0.36 / 0.54 GB | 2.0 / 0.46 / 1.2 GB |

Plaintext first (`results/2026-10-11/wide_plain.py`): the wide model trains like the small one under the
same filter (0.857 vs 0.867, sign-flip 0.880 vs 0.870) with the same honest update norms, so B = 1 fits.
Round time grows slower than the model: training and fixed costs don't scale with it. Memory is what broke
first. (1) The board kept every round forever: about 8 GB over 30 rounds at this size. It now drops the
bulky messages of rounds 2 or more back, and participants may write only from one round back, so the
board stays at ~1.2 GB. (2) A member needs ~75 MB more per ciphertext per update (1.05 GB at 1, 1.1 at 2,
2.0 at 14). On separate machines that is fine. Here 14 parties need 10.3 GB together: the run was killed by
the Docker VM's 8 GB cap and passed with 12 GB (`results/2026-10-11/wide_*.log`). Extrapolated, not
measured: at 1M parameters (123 ciphertexts) a member would need ~10 GB and the board would relay ~7 GB a
round, so models that size need members that stream posts instead of holding them, or a smaller update.

## Related work

From a literature search in September 2026. It isn't exhaustive: the full texts of ShieldFL, PBFL and SAFEFL weren't read.

| work | what it has | compared with this project |
|---|---|---|
| [ByITFL](https://arxiv.org/abs/2405.08698) (Xia, Hofmeister, Egger, Bitar 2024) | FLTrust with information-theoretic privacy (secret sharing); ReLU replaced by a degree-6 polynomial | Users must normalize, and the norm is checked on shares to keep inputs in the polynomial's range. That's gap B's 1-bit gate; the attack without it isn't shown there. They also size the field so nothing wraps. |
| [Bendoukha et al. 2026](https://arxiv.org/abs/2602.05410) | Threshold CKKS with an encrypted Byzantine filter (Chebyshev degree 32, Newton division) | The closest HE system. The server's reference data is public, and inputs outside the approximation range aren't analyzed (gap B may apply). |
| [Lancelot](https://arxiv.org/abs/2408.06197) (2024) | CKKS Krum and median, 10–100 clients, up to ResNet-50 | No approximations: a trusted key center decrypts the distances and sorts them in plaintext. |
| [CrypTFed](https://github.com/Akram275/cryptfed) (open source, TensorFlow/Keras + OpenFHE; read from its README 2026-10-15) | A benchmarking framework: CKKS/BFV/BGV, single-key and threshold; Krum/MultiKrum, FLAME, FoolsGold, trimmed mean; sign-flip, gradient-ascent, noise and label-shuffle attackers | The closest open-source code. Its Krum example sets `use_fhe=False` ("plaintext for robust aggregation"), so the robust rules appear to see plaintext updates; FHE is marked on FedAvg and trimmed mean only. It runs in one process (an orchestrator calling clients), with no range proofs or networked deployment described. This project's difference: the filter runs on ciphertexts with a secret reference, plus range proofs and a deployable signed board. |
| NoV (CCS 2024), ELSA, RoFL, [PINE](https://arxiv.org/abs/2311.10237) | ZK or validity proofs of norm bounds; only an accept bit is revealed | Less leakage than our 2 scores, but the reference isn't private or there's no angle check (PINE alone fails, see "Evaluation"). |
| [SAFEFL](https://eprint.iacr.org/2023/555) | All of FLTrust in MP-SPDZ | The basis for the planned secret-sharing prototype. |
| POSEIDON / SPINDLE, xMK-CKKS | Multiparty or multi-key CKKS training | No poisoning filter. |

**Gap D is known**, including in threshold form: Li–Micciancio 2021; [Checri et al., Crypto 2024](https://eprint.iacr.org/2024/116) (threshold CKKS is insecure without smudging noise after partial decryption); [Guo et al., USENIX Security 2024](https://usenix.org/conference/usenixsecurity24/presentation/guo) (noise flooding sized from average-case estimates is breakable); [Morona-Mínguez et al. 2026](https://arxiv.org/abs/2602.22037) (library-default threshold decryption noise isn't enough against a restricted decryption oracle in FL averaging). That last one prompted a check here: our partial-decryption noise was ~34 bits too small (scale-2¹⁰⁰ decryptions), now replaced by flooding sized with OpenFHE's rule (see the threshold section); Guo et al.'s point still applies to the `u·u` scores, whose error a participant can inflate. **Gap C** is the decryption-oracle (CPAD) model applied to FL. **Gap B:** the defense is known (ByITFL's normalization check), and we found no paper that demonstrates the attack. **Gap F:** the closest idea is PINE's wraparound check in finite fields; we found no CKKS packed-score overflow. **The combination** (threshold CKKS with any committee size and t-of-n, an FLTrust-style filter on ciphertexts, a secret reference held by the committee, a signed untrusted board, deployable) wasn't found elsewhere, and neither was the `eval_ss.py` result that a known reference lets attackers hide a class completely.

## Next steps
- Gaps H and I (2026-10-11): fixed, see above.
- Minimum participants vs. privacy (2026-10-11, `results/2026-10-11/`). With 3 of 10 participants able to withhold,
  the old minimum of 8 let them stall every round; the default is now n − f − 1 = 6 with a round timeout. But the
  minimum is a weak privacy floor: what's decrypted is the sum of the updates the filter keeps, which can be 2.
  If 3 colluders send honest-looking updates (kept like anyone's) and a malicious board shrinks the roster, a
  decrypted round holds exactly one honest update next to theirs in 7% of rounds with a roster of 8, 10% with 7,
  19% with 6 (CNN, 10 runs each, every run had such rounds), which the colluders can approximately subtract. A
  kept floor high enough to rule that out (≥ 5) mostly stops training: the untrained model keeps only ~4 updates,
  so round 0 is skipped forever (floors 5/6: 2–6 of 10 runs never train; under attack worse). So instead the
  members add Gaussian noise of std z·B to each round's sum (`--dp-z`, default 0.1; B = the norm bound, which also
  bounds one participant's effect on the sum). Measured (`dp_measure.py`, CNN, 10 runs per cell):

  | z | acc, no attack | sign-flip ×10 | inside-flip | isolation roster | colluders' reconstruction cosine, median / max | ε per round (δ = 1e-5) |
  |---|---|---|---|---|---|---|
  | 0 | 0.839 | 0.810 | 0.789 | 0.832 | 1.000 / 1.000 | – |
  | 0.01 | 0.833 | 0.801 | 0.788 | 0.838 | 0.210 / 0.394 | ~480 |
  | 0.1 | 0.840 | 0.817 | 0.828 | 0.838 | 0.026 / 0.047 | 48 |
  | 0.14 | 0.837 | 0.836 | 0.824 | 0.831 | 0.020 / 0.035 | 35 |
  | 0.17 | 0.829 | 0.822 | 0.831 | 0.809 | 0.016 / 0.041 | 28 |
  | 0.3 | 0.833 | 0.815 | 0.794 | 0.742 | 0.010 / 0.027 | 16 |
  | 1.0 | 0.691 | 0.563 | 0.501 | 0.590 | 0.007 / 0.018 | 4.8 |

  At 0.1 the lone update is no longer readable (cosine ≤ 0.05 vs 1.0) at no measured accuracy cost. This is a
  mask, not a DP guarantee: ε ≈ 48 per round is meaningless as a bound, and low cosine only rules out direct
  reconstruction, not e.g. membership inference. A real ε (z ≈ 1) costs 15–30 points here. Each member adds the
  full std z·B (not z·B/√members), like the flooding: one honest member's noise is enough, even if the others
  collude with the participants. With 3 members the sum gets 0.17 (0.14 after a member drops): ~3 points on the
  smallest rosters (isolation column), within run-to-run spread elsewhere (`dp_full_measure.log`).
- Gap D for `u·u`: done (`range_proof.py`, `deploy.py --norm-bound`, on by default). Left: PINE is pure Python, but no longer the bulk of a round (CNN rounds 9.9 s median, were 18.6 s), A member dropping out mid-round skips that round on purpose: finishing it would decrypt the same posted ciphertexts a second time (the gap C oracle, and two flooded decryptions to average, gap D); the next round starts at once with fresh posts, so the cost is one round. `TLS=1 NORM_BOUND=1` (2026-10-11): passes, 0.936, 140 s, liar rejected rounds 0–4.
- Member and lead dropouts at any point, and rejoining: done (see Dropouts). 2026-10-12: the successor dying alone hung the run (found by `CRASH=successor-mid`; fixed in need(), see OVERNIGHT_2026-10-12.md). Lead and successor dying together (4 members, threshold 2): the members survive it; a missing-model stall it exposed is fixed; both variants pass (rerun 2026-10-14, see Dropouts).
- Honest rejection is fixed enough to stop costing rounds (clip to the median, see gap E), but ~1/3 of honest non-IID updates still fail the angle check, and about 1 run in 20 dips to 0.63–0.67 under the filter, likely partly its own class swing (repeated draws, gap E). Per-class references were tried as a second score and fail (sign-flips get through). Damping the swing (server momentum) was tested pre-registered on 1000 paired runs and failed (gap E): the unattacked dip is 0.8% of runs. That's the remaining limit, documented, not pursued further.
- Reveal less than 2 scores per participant *to the committee members* (the board no longer sees them). Exact arithmetic (BFV, MPC shares) is needed for a true 1-bit comparison, but there a malicious participant can wrap its norm mod p (gap F), so every input needs a validity proof. Evaluated (see above): secret sharing + PINE with B = |ref| + one MPC comparison for the angle keeps the protection and reveals 2 bits. Prototyped 2026-10-13 (above): it works, but malicious MPC costs ~20× the CKKS scoring traffic and a member can lie about its share, so the CKKS version stays. Deployability is done: multiple machines, signed board messages, config file, any committee size ≥ 2 and any number of participants. Member dropouts are done (t-of-n, above) except for the lead. Left there: no pip package (the Docker image is the package, since the `openfhe` wheel is Linux-only), and it has run across two machines only as a WSL VM next to Docker on one laptop (see Deployment), not on separate hardware over the internet.

## Future scope: a website training on its visitors' data
Cross-device setting: many short-lived browsers with little data each, where the website backend must learn the model but not any visitor's update. Suggested design, reusing the PINE path above instead of CKKS (browser ciphertexts are MBs, and there are no trusted key holders):
- Browser trains a small model in JS/WASM, clips the update, splits it into two secret shares plus a PINE norm proof, and sends one share to the website and one to an independent helper (second org, or a TEE if hardware trust is acceptable). The servers check proofs, sum shares, and combine only the sums. Dropouts cost nothing: a closed tab is just missing from the sum.
- Threats that need their own fixes: (1) the site serves the JS, so it could ship code that exfiltrates raw data. Crypto can't fix that; the client must be pinned and auditable (extension, app binary, hash in a transparency log). (2) Sybil / n-1: the site fills a batch with fakes around one victim, so the helper enforces a minimum batch size. (3) Memorisation: add user-level DP noise to the aggregate.
- Demo path: Python "browsers" on the existing deploy/board first, then a JS client; task = click prediction with LR (tasks/lr.py); dashboard shows the model improving while site logs hold only random shares, plus a Sybil toggle.
