#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data for snax-dsv2-gemm3-bench: GEMMs of HeMAiA's MoE prefill on VersaCore's array shape 3,
# (8, 8, 16), with INT4 weights in one of three layouts and a row-major A (include/snax-dsv2-g3.h).
#
# A case is a GEMM (mt tiles of 8 rows, kt passes of 8, nt tiles of 16 columns), the layout of its
# weights and where B sits against A. A launch case is a run of back-to-back GEMMs of one shape,
# launched one of six ways (src/snax-dsv2-gemm3-bench.c).
#
# Every operand of one depth K comes from one A [32 rows, 8 K] and one weight [8 K, 96] of INT4
# values; a smaller shape reads a prefix of the blobs (rows, and column tiles in all three
# layouts), so a case's expected D is the product of those prefixes.
#
# Per depth K, in DRAM: g3_a<K> (row-major, pitch G3_PITCH), g3_b<K>_<layout>; per shape
# g3_y<K>_<mt>_<nt>, the expected D = RNE(A . B * 2^-k), FP16 in the D-layout.

import argparse
import pathlib
import sys

import hjson
import numpy as np

import g3data as g3  # noqa E402
from util.emit import Emitter  # noqa E402

NT = 6          # column tiles of every case: a gate|up chunk of three blocks each
MT_MAX = 4      # row tiles: up to 32 rows
LAYS = ("b", "paired", "parity")


def case_lists(name, chunk_boffs):
    """(cases, launches): cases (K, mt, nt, layout, boff, A layout), launches (K, mt, nt, layout,
    boff, A layout, reps, kind). The kinds are src/snax-dsv2-gemm3-bench.c's G3B_*."""
    cases, launches = [], []
    if name == "unit":
        cases = [(256, 2, NT, lay, chunk_boffs[lay], "row") for lay in LAYS]
        cases += [(256, 2, NT, "b", 0, "blocked"), (256, 2, NT, "paired", 0, "ilv")]
        launches = [(256, 1, NT, "parity", g3.PARITY_OFF, "row", 4, kind) for kind in range(6)]
    elif name == "scan":
        cases = [(512, 2, NT, lay, boff, "row") for lay in LAYS for boff in range(0, 256, 8)]
    elif name == "scanp":  # the paired layout's best B offset at the chunk depth
        cases = [(2048, 2, NT, "paired", boff, "row") for boff in range(0, 256, 8)]
    elif name == "chunk":
        cases = [(2048, mt, NT, lay, chunk_boffs[lay], "row")
                 for mt in range(1, MT_MAX + 1) for lay in LAYS]
        # the control: B an even number of words past A
        cases += [(2048, mt, NT, "parity", 0, "row") for mt in (1, 2)]
    elif name == "amode":  # A layout x B layout: equal strides keep their banks apart
        for mt in (2, 3):
            cases += [(2048, mt, NT, "parity", g3.PARITY_OFF, "row"),
                      (2048, mt, NT, "paired", 0, "row"), (2048, mt, NT, "b", 0, "row")]
            for lay in ("b", "parity"):
                cases += [(2048, mt, NT, lay, off, "blocked") for off in (0, 8, 64, 128, 192)]
            cases += [(2048, mt, NT, "paired", off, "blocked") for off in (0, 64, 128)]
            for lay in ("paired", "b", "parity"):
                cases += [(2048, mt, NT, lay, off, "ilv") for off in (0, 8, 64, 128, 192)]
    elif name == "launch":
        launches = [(K, mt, NT, "parity", g3.PARITY_OFF, "row", 8, kind)
                    for K in (2048, 256) for mt in (1, 2) for kind in range(6)]
    else:
        raise ValueError(f"no case set {name!r}: unit, scan, scanp, chunk, amode or launch")
    return cases, launches


def main():
    ap = argparse.ArgumentParser(description="Data for snax-dsv2-gemm3-bench")
    ap.add_argument("--swcfg", type=pathlib.Path, required=True)
    ap.add_argument("--hwcfg", type=pathlib.Path, required=True)
    ap.add_argument("--header", type=pathlib.Path, required=True)
    ap.add_argument("--blob-dir", type=pathlib.Path, required=True)
    ap.add_argument("--set", default="unit",
                    help="the cases: unit, scan, scanp, chunk, amode or launch")
    args = ap.parse_args()
    prm = hjson.loads(args.swcfg.read_text())
    g3.check_hwcfg(hjson.loads(args.hwcfg.read_text()))
    k = int(prm["k"])
    boffs = {lay: int(prm["chunk_boffs"][lay]) for lay in LAYS}
    cases, launches = case_lists(args.set, boffs)
    for K, mt, nt, lay, boff, *_ in cases + launches:
        if boff % 8 or not 0 <= boff < 256:
            raise ValueError(f"B offset {boff}: a multiple of 8 below 256")
        if (K // 8) % 2:
            raise ValueError(f"K {K}: an even number of passes")

    rng = np.random.default_rng(int(prm["seed"]))
    em = Emitter(args.blob_dir)
    em.c(f"// snax-dsv2-gemm3-bench, set {args.set}: {len(cases)} cases, {len(launches)} launch "
         f"cases, seed {prm['seed']}.")
    em.define("G3B_K", k, "the D port's RNE(acc * 2^-k)")
    em.define("G3B_REPS", int(prm["reps"]), "runs of each case, the last measured")
    em.define("G3B_NCASES", len(cases))
    em.define("G3B_NLAUNCH", len(launches))

    depths = sorted({c[0] for c in cases + launches}, reverse=True)
    a_of, w_of, b_of, y_of = {}, {}, {}, {}
    for K in depths:
        X = rng.integers(-128, 128, size=(8 * MT_MAX, K)).astype(np.int8)
        W = rng.integers(-8, 8, size=(K, g3.MESH3[2] * NT)).astype(np.int8)
        a_of[K] = (X, {al: em.blob(f"g3_a{K}_{al}", g3.a_blob(X, al, g3.pitch(K)))
                       for al in sorted({c[5] for c in cases + launches if c[0] == K})})
        w_of[K] = W
        for lay in sorted({c[3] for c in cases + launches if c[0] == K}):
            b_of[K, lay] = em.blob(f"g3_b{K}_{lay}", g3.b_blob(W, lay))
    for K, mt, nt, *_ in cases + launches:
        if (K, mt, nt) not in y_of:
            X, W = a_of[K][0], w_of[K]
            y_of[K, mt, nt] = em.blob(f"g3_y{K}_{mt}_{nt}",
                                      g3.d_blob(X[:8 * mt], W[:, :g3.MESH3[2] * nt], k))

    abytes = lambda K, mt, al: 8 * mt * (g3.pitch(K) if al == "row" else K)  # noqa E731
    bbytes = lambda K, nt: K * g3.MESH3[2] * nt // 2  # noqa E731
    rows = []
    for K, mt, nt, lay, boff, al in cases:
        rows.append(f'    {{"k{K}m{8 * mt}{al[0]}{lay[0]}+{boff}", {mt}u, {K // 8}u, {nt}u, '
                    f'{g3.LAYOUTS[lay]}u, {boff}u, {g3.ALAYOUTS[al]}u, {g3.pitch(K)}u, '
                    f'{a_of[K][1][al]}, {b_of[K, lay]}, {abytes(K, mt, al)}u, {bbytes(K, nt)}u, '
                    f'{y_of[K, mt, nt]}}},')
    lrows = []
    for K, mt, nt, lay, boff, al, reps, kind in launches:
        lrows.append(f'    {{"k{K}m{8 * mt}", {mt}u, {K // 8}u, {nt}u, {g3.LAYOUTS[lay]}u, '
                     f'{boff}u, {g3.ALAYOUTS[al]}u, {g3.pitch(K)}u, {reps}u, {kind}u, '
                     f'{a_of[K][1][al]}, {b_of[K, lay]}, {abytes(K, mt, al)}u, {bbytes(K, nt)}u, '
                     f'{y_of[K, mt, nt]}}},')
    amax = max([abytes(c[0], c[1], c[5]) for c in cases + launches])
    bmax = max([bbytes(c[0], c[2]) for c in cases + launches])
    dmax = max([c[1] * c[2] * 256 for c in cases + launches])
    em.define("G3B_ABYTES", amax, "the largest A a case reads")
    em.define("G3B_BBYTES", bmax, "the largest B a case reads")
    em.define("G3B_DBYTES", dmax, "the largest D a case writes, FP16")
    em.c("typedef struct {\n    const char *name;\n"
         "    uint32_t mt, kt, nt;  // tiles: 8 rows, 8 deep, 16 columns\n"
         "    uint32_t lay;         // g3_lay_t\n"
         "    uint32_t boff;        // B's bytes past a 256-byte boundary; A sits on one\n"
         "    uint32_t alay;        // g3_alay_t\n"
         "    uint32_t pitch;       // A's row pitch (row-major)\n"
         "    const int8_t *a, *b;\n    uint32_t abytes, bbytes;\n"
         "    const uint16_t *y;    // the expected D, FP16 in the D-layout\n} g3b_case_t;")
    em.c("typedef struct {\n    const char *name;\n"
         "    uint32_t mt, kt, nt, lay, boff, alay, pitch;\n"
         "    uint32_t reps;        // GEMMs back to back\n"
         "    uint32_t kind;        // G3B_GEN/LEAN/DELTA + G3B_ASYNC\n"
         "    const int8_t *a, *b;\n    uint32_t abytes, bbytes;\n"
         "    const uint16_t *y;\n} g3b_launch_t;")
    em.c("static const g3b_case_t g3b_cases[G3B_NCASES + 1] = {\n" + "\n".join(rows)
         + "\n    {0}};")
    em.c("static const g3b_launch_t g3b_launch[G3B_NLAUNCH + 1] = {\n" + "\n".join(lrows)
         + "\n    {0}};")
    em.blob("g3_nan", np.full(max(dmax, 256) // 2, 0x7E00, dtype=np.uint16))
    em.write(args.header, args.blob_dir / "blobs.S")
    sys.stderr.write(f"[gemm3-bench datagen] set {args.set}: {len(cases)} cases, {len(launches)} "
                     f"launch cases, {em.bytes / 2**20:.2f} MiB of blobs\n")


if __name__ == "__main__":
    main()
