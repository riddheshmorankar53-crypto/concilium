#!/bin/sh
# Builds the concilium image once (10-20 min, ~2 GB): every package is pinned by hash, so all parties get the same.
cd "$(dirname "$0")"
docker build --target tool -t concilium . && sh concilium.sh version && echo "Done. Compare the code hash above with the other parties."
