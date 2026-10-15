#!/bin/sh
cd "$(dirname "$0")"
docker build --target tool -t concilium . && echo Done.
