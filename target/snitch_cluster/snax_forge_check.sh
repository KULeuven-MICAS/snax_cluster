#!/usr/bin/env bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# snax-forge-check (snax_forge.mk): each snax-forge bundle of BUNDLES, in order, on this cluster's
# RTL, compared with the bundle's own SNAX-GVSoC run.
#
#   BUNDLES             the bundles' folders, absolute, made by snax-forge's `pixi run forge`
#                       (out/bundle/<name>/, with forge_expected.txt beside its files)
#   SIM                 vsim or vlt
#   SNAX_CLUSTER_IMAGE  the development image (podman) for the steps run in it; required
#
# Per bundle B (its name, B's last folder), the commands snax-forge's forge prints for it, with
# ELF the program snax-forge built from the bundle (out/forge/<name>/sw/forge.elf beside
# out/bundle/<name>/), after `clean-generated clean-vlt clean-vsim`:
#
#   vsim  rtl-gen and vsim_preparation in the image, then snax-forge-sim and snax-forge-run
#         outside it (bender, here snax_cluster's pixi environment's, and Questa on PATH)
#   vlt   all four in the image
#
# podman runs non-interactively, snax_cluster, B and the ELF's folder mounted at their own
# paths. Then the run's `forge counter` lines, in order, and its `forge data:` line must equal
# B/forge_expected.txt: `PASS <name>` or `FAIL <name>: <want> / <got>`. Each design's whole
# output is kept in logs/snax-forge-<name>-<sim>.txt, the PASS and FAIL lines and each run's
# `forge cycles:` line in logs/snax-forge-check.txt. Exit 1 when a design failed, 2 on bad
# arguments.

set -u
HERE=$(cd "$(dirname "$0")" && pwd)   # target/snitch_cluster
REPO=$(cd "$HERE/../.." && pwd)
LOGS=$HERE/logs
CHECK=$LOGS/snax-forge-check.txt

die() { echo "snax-forge-check: $*" >&2; exit 2; }
[ -n "${BUNDLES:-}" ] || die "BUNDLES is not set: the bundles' folders, absolute (snax-forge's forge-set prints them)"
case "${SIM:-}" in vsim|vlt) ;; *) die "SIM=${SIM:-}: use vsim or vlt";; esac
[ -n "${SNAX_CLUSTER_IMAGE:-}" ] || die "SNAX_CLUSTER_IMAGE is not set: the development image's id (podman images), e.g. c5db84ad7bce"
if [ "$SIM" = vsim ]; then
    command -v vsim > /dev/null || die "no vsim on PATH (Questa)"
    command -v bender > /dev/null || PATH=$PATH:$REPO/.pixi/envs/default/bin
    command -v bender > /dev/null || die "no bender on PATH, nor in $REPO/.pixi/envs/default/bin"
fi

mkdir -p "$LOGS"
: > "$CHECK"
failed=0

# in the image, from target/snitch_cluster, with the folders it reads mounted
in_image() {
    local mounts=(-v "$REPO:$REPO" -v "$B:$B" -v "$(dirname "$ELF"):$(dirname "$ELF")")
    podman run --rm "${mounts[@]}" -w "$HERE" "$SNAX_CLUSTER_IMAGE" bash -lc "$1" < /dev/null
}
here() { (cd "$HERE" && bash -c "$1") < /dev/null; }

report() {  # report LINE: to the terminal and to the check file
    echo "$1"
    echo "$1" >> "$CHECK"
}

for B in $BUNDLES; do
    B=${B%/}
    name=$(basename "$B")
    log=$LOGS/snax-forge-$name-$SIM.txt
    run_log=$LOGS/snax-forge-$SIM.txt
    CFG=$B/snax_forge.hjson
    ELF=$(dirname "$(dirname "$B")")/forge/$name/sw/forge.elf
    if [ ! -f "$B/forge_expected.txt" ]; then
        report "FAIL $name: no $B/forge_expected.txt: make the bundle with snax-forge's pixi run forge (or forge-set), not pixi run bundle"
        failed=1; continue
    fi
    if [ ! -f "$CFG" ] || [ ! -f "$ELF" ]; then
        report "FAIL $name: no $CFG or no $ELF: make the bundle with snax-forge's pixi run forge"
        failed=1; continue
    fi
    echo "== $name: $SIM, CFG_OVERRIDE=$CFG ELF=$ELF; its output in $log"
    clean="make clean-generated clean-vlt clean-vsim && rm -f $run_log"
    gen="make rtl-gen CFG_OVERRIDE=$CFG"
    sim="make snax-forge-sim SIM=$SIM CFG_OVERRIDE=$CFG"
    run="make snax-forge-run SIM=$SIM CFG_OVERRIDE=$CFG ELF=$ELF"
    {
        if [ "$SIM" = vsim ]; then
            in_image "$clean && $gen && make vsim_preparation CFG_OVERRIDE=$CFG" && here "$sim && $run"
        else
            in_image "$clean && $gen && $sim && $run"
        fi
    } > "$log" 2>&1
    rc=$?
    want=$(cat "$B/forge_expected.txt")
    got=$( { grep -o 'forge counter .*' "$run_log"; grep -o 'forge data: .*' "$run_log"; } 2> /dev/null)
    cycles=$(grep -o 'forge cycles: .*' "$run_log" 2> /dev/null | head -n 1)
    if [ $rc -ne 0 ]; then
        report "FAIL $name: a step failed (exit $rc), see $log"
        failed=1
    elif [ "$want" = "$got" ]; then
        report "PASS $name"
    else
        report "FAIL $name: ${want//$'\n'/; } / ${got//$'\n'/; }"
        failed=1
    fi
    echo "  $name ${cycles:-forge cycles: none}" >> "$CHECK"
done

echo "the lines above and each run's forge cycles: $CHECK"
exit $failed
