# Helping with a test run: you are "member 1"

Thank you for helping. Your computer will be one of three committee members in a privacy-preserving
machine-learning test. It holds one piece of a shared key and helps decrypt only the final, combined
result. It never sees anyone's data, and it doesn't touch your files: everything runs inside Docker, in
this folder.

**You need:** Docker Desktop (free for personal use), about 5 GB of free disk, about 4 GB of free memory
during the run, and about an hour (mostly downloads). Your internet will download about 2 GB once (the
software) and about 1 GB during the run.

## Once, before the test

1. **Install Docker Desktop** from docker.com and start it (wait until it says it's running).
2. **Only if you're not on the same Wi-Fi: install Tailscale** from tailscale.com and sign in (free). Your friend will "share" their laptop with
   your Tailscale account. Accept the invitation when it arrives. This lets your computer reach only their
   laptop, nothing else, and their laptop can't reach yours.
3. **Unzip:** right-click the zip, choose "Extract All", and open the extracted folder. Everything below
   happens in that folder (the scripts don't work from inside the zip).
4. **Build the software:** double-click `1_build.cmd` (Mac/Linux: run `sh 1_build.sh` in a terminal in this
   folder). It downloads and builds for 10–20 minutes. It's done when it says "Done".
5. **Make your keys:** double-click `2_keys.cmd` (Mac/Linux: `sh 2_keys.sh`). It creates two files:
   - `m1.key`: your private key. **Never send it to anyone.** It stays in this folder.
   - `m1.pub`: your public key. **Send this one file to your friend.**

## On the day

6. Your friend sends you a file `concilium.json` and their laptop's address: on the same Wi-Fi its local IP
   (like `192.168.1.2`), over the internet its Tailscale address (starts with `100.`). Put `concilium.json` in
   this folder.
7. Double-click `3_run.cmd` (Mac/Linux: `sh 3_run.sh`) and paste that address when asked.
8. Leave the window open. It prints `member 1, session ...` and then works quietly. A run takes about
   5–15 minutes. When it ends, the window says the run finished. Send your friend a screenshot of that window.

If it stops with an error, send your friend a screenshot. Nothing on your computer is harmed: you can
delete this folder and, in Docker Desktop, the image called `concilium`.

## Afterwards

- In Tailscale, you can remove the shared laptop (or your friend removes the share).
- Delete this folder and the `concilium` image in Docker Desktop if you want the space back. Don't send
  `m1.key` or `m1.key.state` to anyone: they're private and only matter for this test.
