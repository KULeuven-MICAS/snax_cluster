#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Build snax_csr_range_demux_tb with Verilator (--binary, timing on) and run it once per
# seed (default 1 2 3). Exits nonzero unless every seed prints PASS.
# Usage: hw/snax_util/test/run_snax_csr_range_demux_tb.sh [seed ...]
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD=${BUILD:-$(mktemp -d)}
SEEDS=${*:-1 2 3}
mkdir -p "$BUILD"
verilator --binary --timing -j 0 -Wno-fatal --top-module snax_csr_range_demux_tb \
  --Mdir "$BUILD" -o Vtb \
  "$HERE/../src/snax_csr_range_demux.sv" "$HERE/snax_csr_range_demux_tb.sv" \
  > "$BUILD/build.log" 2>&1 \
  || { tail -30 "$BUILD/build.log"; exit 1; }
status=0
for seed in $SEEDS; do
  line=$("$BUILD/Vtb" +seed="$seed" | grep -E "^(PASS|FAIL)" || echo "FAIL seed=$seed: no verdict")
  echo "$line"
  case "$line" in PASS*) ;; *) status=1 ;; esac
done
exit $status
