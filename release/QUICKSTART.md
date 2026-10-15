# Concilium: train one model on several parties' data, without sharing the data

Each participant trains on its own data and sends only an encrypted model update. A committee of members
holds pieces of the decryption key: together they decrypt only the filtered sum of the updates, never one
participant's update, and they reject poisoned updates. A board (a plain server nobody has to trust)
relays the messages. Overview: README.md; threat model, attacks and measurements: RESEARCH.md.

Commands below use `sh concilium.sh ...` (Mac/Linux). On Windows type `concilium ...` in a terminal opened in this folder.

## Roles

| role | what it does | runs | needs (measured) |
|---|---|---|---|
| organizer | collects public keys, writes `concilium.json` | `config` once | nothing special |
| board | relays signed messages; untrusted | `board` for the whole run | ~1 GB memory, a port everyone can reach |
| member (2+, usually 3) | holds a key piece, scores and decrypts | `member` for the whole run | 1–2 GB memory up to ~100k parameters, a few cores |
| participant | trains on its own data | `participant` for the whole run | 0.5 GB memory plus what its model needs |

One machine may hold several roles, but members should be different organizations: any t of them
together could decrypt single updates (t = members − 1 by default).

## Once, everyone

1. Install Docker (Docker Desktop on Windows/Mac) and start it. Unzip this folder.
2. Get the image, one of:
   - Download (~1.8 GB): `docker pull ghcr.io/riddheshmorankar53-crypto/concilium:1c669e7960649da1`, then
     `docker tag ghcr.io/riddheshmorankar53-crypto/concilium:1c669e7960649da1 concilium`. It was built from this
     code by GitHub Actions; building it yourself gives the same hash.
   - Build it yourself from this folder: `sh build.sh` (Windows: `build.cmd`). 10–20 minutes.

   Then `sh concilium.sh version` (Windows: `concilium version`) prints `concilium code <hash>`. **Compare that hash with the
   other parties** (phone, chat): everyone must run the same code.
3. Make your keys: `sh concilium.sh keys <name>`, e.g. `keys m0` for a member or `keys p3` for a participant.
   `<name>.key` stays in this folder: never send it. Send `<name>.pub` to the organizer.

## The organizer

4. Agree on a task file with everyone: what the data and the model are.
   - **Tabular data in CSV files, yes/no prediction: no coding.** Every party copies `tasks/csv_task.py` into
     this folder. Participants put their rows in `my_data.csv` next to it, members a small clean labeled set in
     `root.csv` (and optionally `test.csv`). Everyone sets the same `LABEL` (the 0/1 column) at the top of the
     file; the organizer runs `docker run --rm -v "$PWD:/work" concilium python3 csv_task.py stats root.csv`
     (Windows: `"%cd%:/work"`) and sends everyone the `MEAN` / `STD` lines it prints, to paste in. Then use `--task csv_task.py` below and `TASK=csv_task.py` when running.
   - Anything else: copy `tasks/lr.py` or `tasks/cnn.py` and fill in its functions (the contract is in
     `tasks/lr.py`). Each party keeps its copy in its folder, with its own data paths in `participant_data`
     (participants) or `root_data` (members: a small clean labeled set).
5. Write the config:

   ```
   sh concilium.sh config concilium.json --board https://board.example.org:8000 --task my_task.py \
       --committee m0.pub m1.pub m2.pub --participants p0.pub p1.pub p2.pub ...
   ```

   The first member listed leads. Range proofs are on with `--norm-bound 1`: pick the bound from your
   task's honest update norms. `sh concilium.sh config --help` lists the rest (round minimum, timeouts, noise).
6. Send `concilium.json` to everyone. It holds only public keys. Every party prints a **session id** when it
   starts: compare it over a second channel. It covers the config and the code, so a different file or a
   different build shows up as a different id.

## The run

7. Board: get a TLS certificate for the board's name (`cert.pem`, `key.pem`; from your CA or e.g. Let's
   Encrypt), then `sh concilium.sh board concilium.json cert.pem key.pem`. Without the two files it serves plain
   HTTP: only for tests (anyone on the path can watch and replay traffic).
8. Members, then participants: `sh concilium.sh member concilium.json m0.key` and
   `sh concilium.sh participant concilium.json p3.key`. With your own copy of the task file:
   `TASK=my_task.py sh concilium.sh participant concilium.json p3.key` (Windows: `set TASK=my_task.py` first).
9. Leave them running. The lead member prints the result at the end. A member that hits an error stops
   the run for everyone; a member that just disappears is dropped and the others go on (with the default
   threshold, one may), and it can rejoin by running the same command again. A participant that drops is
   simply left out of later rounds.
10. If too many members disappear at once (fewer than the threshold left), nobody can decrypt any more: the last
   members stop the run for everyone. To go on from where it stopped, while the old board is still up:
   `sh concilium.sh export concilium.json model.bin` (anyone; it takes the latest model enough members signed), then the
   organizer writes a new config with `--init-model model.bin` added and everyone starts again from step 7.
   To survive k members disappearing together, use at least threshold + k members: e.g. 4 members with
   `--threshold 2` survive 2, but then any 2 of them together could decrypt single updates.

To try it all on one machine first: `--task lr` is a built-in example on public data (breast cancer, via
scikit-learn); `--task cnn` needs `pneumoniamnist.npz` (MedMNIST) in this folder.

## Good to know

- A board with a private CA: put the CA certificate in this folder and add `-e SSL_CERT_FILE=/work/ca.pem`
  to the `docker run` line in `concilium.sh` / `concilium.cmd`.
- Slow links: `NET_TIMEOUT=60 sh concilium.sh member ...` (seconds per request; default 5).
- Model size: rounds took 6 s (31 parameters), 10 s (14k) and 42 s (108k) with 10 participants; upload
  per participant per round 1.3, 2.6 and 18.6 MB.
- Nothing here is audited. RESEARCH.md ("Known limits, read before deploying") lists what is still open; read it before relying on the privacy.
