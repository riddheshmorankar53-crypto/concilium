# Concilium: train one model on several parties' data, without sharing the data

Several organizations (hospitals, banks, companies, anyone with data they can't hand over) train one shared
model together. Each trains on its own data and sends only an **encrypted** model update. A small committee
holds pieces of the decryption key: together they decrypt only the combined, filtered result, never one
party's update, and they **reject poisoned updates** without seeing them. A plain server relays the messages
and is not trusted.

**Status:** a working research prototype, not audited. It runs end to end (Docker, 14 parties, two real
laptops), every claim below is checked by a script, and the known gaps are listed. Don't protect truly
sensitive data with it before a security review.

## What it does differently

Federated-learning frameworks usually give you either encrypted aggregation (the server can't read updates)
or robust aggregation (the server filters out poisoned updates), not both, because a filter has to look at
the updates. Here both happen together:

- **Threshold CKKS encryption** (OpenFHE): no single key holder; any t of n committee members decrypt, and the
  run survives members dropping out, rejoining and the lead failing over.
- **A poisoning filter on ciphertexts** (FLTrust-style: angle to a reference update, length clipped to the
  median), with the reference data kept secret by the committee.
- **Range proofs** (PINE): every participant proves its update is bounded, so oversized or forged updates
  can't break the filter or the decryption noise.
- **A signed, untrusted board:** every message is signed and bound to the session; the board can delay or drop
  messages, not forge or read them.
- **Attacks it was built against**, each reproduced and fixed by a script: approximation attacks on encrypted
  FLTrust, the key holder as a decryption oracle, key recovery from exact CKKS decryptions, packed-score
  overflow, optimizing attackers, participant-triggered aborts, malleable ciphertexts. See
  [RESEARCH.md](RESEARCH.md).

## Try it on one machine (about 5 minutes)

Needs Docker and Python 3 with `pip install cryptography numpy scikit-learn` (the containers do the rest).

```
git clone <this repository> && cd <it>
docker pull ghcr.io/riddheshmorankar53-crypto/concilium:latest
IMAGE=ghcr.io/riddheshmorankar53-crypto/concilium:latest python deploy.py
```

This starts a board, 3 committee members and 10 participants as separate containers on breast cancer data,
with 3 attackers and one participant lying in its range proof. The lead prints the accuracy (about 0.91-0.95),
what was rejected, and the traffic. `python deploy.py --help` lists the real-deployment commands.

## Use it with your own data

[release/QUICKSTART.md](release/QUICKSTART.md) walks every role through a real deployment, one machine per
party. For tabular data in CSV files (yes/no prediction) no coding is needed: `tasks/csv_task.py`. Other
models: copy `tasks/lr.py` (logistic regression) or `tasks/cnn.py` (PyTorch CNN) and fill in five functions.
Everything can run at no cost: the image is public, and the board can run on one party's own computer.

## Limits, in short

- **Trust:** fewer than t committee members collude, and members follow key generation honestly.
- **Members see** two scores per participant per round, not the updates.
- **Poisoning is bounded, not prevented**; on skewed data about 1/3 of honest updates are left out of a round.
- **The added noise is a mask, not differential privacy** (ε ≈ 48 per round).
- **Size:** up to ~100k parameters on CPUs; 10 participants per run were tested.
- **Tested on** Windows + Docker Desktop and two laptops on one Wi-Fi; not yet on Mac/Linux hosts or across the internet.

The full list, with measurements: RESEARCH.md, "Known limits, read before deploying".

## Repository

| | |
|---|---|
| `deploy.py` | the tool: keys, config, board, member, participant, and the local demo |
| `threshold.py`, `fl_core.py`, `range_proof.py` | threshold CKKS, the filter, the range proofs |
| `gap_*.py`, `flood_probe.py`, `ss_mpc.py`, ... | the attacks and evaluations behind RESEARCH.md; each asserts its own claims |
| `demo.py` | runs all 15 steps of the story in order (~30 min) |
| `tasks/` | example tasks and the CSV template |
| `release/`, `friend/` | the packages for real deployments |

License: MIT.
