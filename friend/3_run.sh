#!/bin/sh
cd "$(dirname "$0")"
[ -f concilium.json ] || { echo "Put the concilium.json your friend sent into this folder first."; exit 1; }
printf "Paste the address your friend sent (e.g. 192.168.1.2, or 100.x.y.z with Tailscale):"; read IP
docker rm -f concilium-member >/dev/null 2>&1
docker run --rm --name concilium-member --user "$(id -u):$(id -g)" -e HOME=/tmp --add-host "concilium-board:$IP" -e NET_TIMEOUT=60 -e OMP_NUM_THREADS=4 -e OMP_WAIT_POLICY=PASSIVE -v "$PWD:/work" concilium concilium member concilium.json m1.key && echo "The run finished. Send your friend a screenshot." || echo "The run stopped with an error. Send your friend a screenshot."
