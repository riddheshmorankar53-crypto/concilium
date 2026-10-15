#!/bin/sh
cd "$(dirname "$0")"
[ -f m1.key ] && { echo "m1.key already exists: keeping it."; exit 0; }
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/work" concilium concilium keys m1 && echo "Send ONLY the file m1.pub to your friend. Never send m1.key."
