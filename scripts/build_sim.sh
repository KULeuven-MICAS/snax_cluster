#!/bin/bash
# Bundle makefile commands to build RTL, software and simulator.
# Supports running in and outside of the SNAX container.


set -e


ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

bender update --fetch

podman run --rm -i -v "$(pwd)":"$(pwd)" -w "$(pwd)" ghcr.io/kuleuven-micas/snax:main bash -s <<'IN_CONTAINER'
set -e
cd target/snitch_cluster
make CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson vsim_preparation
make CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson rtl-gen-no-rebuild
make CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson sw -j
IN_CONTAINER

# Make simulator
cd "${ROOT_DIR}/target/snitch_cluster"
make CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson bin/snitch_cluster.vsim 



