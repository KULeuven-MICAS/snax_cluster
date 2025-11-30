#!/bin/bash
# Bundle makefile commands to build RTL for synthesis.
# Supports running in and outside of the SNAX container.


set -e


ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

bender update

podman run --rm -i -v "$(pwd)":"$(pwd)" -w "$(pwd)" ghcr.io/kuleuven-micas/snax:main bash -s <<'IN_CONTAINER'
set -e
cd target/snitch_cluster
make CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson rtl-gen
IN_CONTAINER

# Make flists
mkdir -p scripts_generated
bender script synopsys -t tech_cells_generic_exclude_tc_sram -t synthesis -t snax_simbacore -t snax_simbacore_cluster  > scripts_generated/flist-dc.tcl
bender script genus -t tech_cells_generic_exclude_tc_sram -t synthesis -t snax_simbacore -t snax_simbacore_cluster  > scripts_generated/flist-genus.tcl

