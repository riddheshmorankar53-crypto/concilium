#!/bin/sh
# One concilium command, run in Docker on this folder: sh concilium.sh keys|config|board|member|participant|version ...
# Your keys, the config, your data and task file stay in this folder; the container sees only this folder.
cd "$(dirname "$0")"
port=""
[ "$1" = board ] && port="-p ${PORT:-8000}:${PORT:-8000}"  # the board must be reachable by everyone
# As you, not root: on Linux, files the container writes (your keys!) would otherwise belong to root
exec docker run --rm $port --user "$(id -u):$(id -g)" -e HOME=/tmp -e TASK -e NET_TIMEOUT -e PORT -e OMP_WAIT_POLICY=PASSIVE -v "$PWD:/work" concilium concilium "$@"
