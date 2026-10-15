# OpenFHE's pip wheel is Linux-only (built for Ubuntu 24.04), so the tool and the threshold experiments run here.
#   docker build -t concilium-research .                       research image (default): the tool + TenSEAL for the gap scripts
#   docker build --target tool -t concilium .    what a party needs to deploy: the tool only
#   docker run --rm -v "${PWD}:/work" concilium-research python3 threshold.py
# Pinned: the base image by digest, every Python package by version and file hash (requirements*.txt), so two
# parties building months apart get the same packages, and a changed upload fails the build. apt packages
# (python3, pip, libgomp1) come from the pinned Ubuntu release's archive.
FROM ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55 AS tool
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/
# torch's CPU build comes from PyTorch's index; everything else from PyPI
RUN pip3 install --no-cache-dir --break-system-packages --require-hashes --no-deps -r /tmp/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
WORKDIR /work
# The tool itself, so a party needs only this image and a folder of its own (keys, config, data, task file):
#   docker run --rm -v "${PWD}:/work" concilium concilium keys alice
# Scripts run from a mounted source folder (python3 deploy.py, the demo) still use that folder's code.
# deploy.CODE lists the same files: their hash is part of every session id (concilium version prints it).
COPY deploy.py threshold.py fl_core.py gap_c.py gap_b.py gap_b_cnn.py range_proof.py /app/
COPY tasks /app/tasks
COPY pine_poc /app/pine_poc
RUN printf '#!/bin/sh\nexec python3 /app/deploy.py "$@"\n' > /usr/local/bin/concilium && chmod +x /usr/local/bin/concilium

FROM tool AS research
COPY requirements-research.txt /tmp/
RUN pip3 install --no-cache-dir --break-system-packages --require-hashes --no-deps -r /tmp/requirements-research.txt
