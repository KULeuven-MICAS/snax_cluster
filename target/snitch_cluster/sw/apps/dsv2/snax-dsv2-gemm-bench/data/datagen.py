#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data for snax-dsv2-gemm-bench: GEMM shapes in isolation on the (16, 4, 16) array, the shapes
# of HeMAiA's FFN prefill GEMM bench (workloads/ffn/common/gemm_bench_mc.py there).
#
# A case is a shape (M row tiles of 16, K passes of 4, N column tiles of 16), the way B is laid
# out and read (the mode), and where B sits against A. The modes:
#   i8    INT8 weights in the B-layout: 64 B a pass, tile n at 64 K n
#   i4    INT4 weights in the paired layout (util/layout.py to_b_pairs, pack_int4): 32 B a pass,
#         consecutive passes of a block 64 B apart, as A's (snax-dsv2.h, THE GEMM)
#   i4b   INT4 weights in the B-layout, nibble-packed: 32 B a pass, 32 B apart -- HeMAiA's GEMM
#         kernel's layout, the bank-conflict control
#
# Every operand of one depth K comes from one A [16 Mmax, 4 K] and one weight [4 K, 16 Nmax] of
# INT4 values, so a smaller shape reads a prefix of the blobs (all three layouts are tile-major)
# and the three modes of one shape compute the same D: one golden per shape.
#
# Per depth K, in DRAM: gb_a<K> (A-layout), gb_b<K>_<mode> per mode used; per shape gb_y_<name>,
# the expected D = RNE(A . W * 2^-k), FP16 in the D-layout.

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))  # the util package
from util.emit import Emitter  # noqa E402
from util.fp import bits16, d_port  # noqa E402
from util.layout import mesh_from_hwcfg, pack_int4, to_a, to_b, to_b_pairs, to_d  # noqa E402

# name: (M, K, N) in tiles. The prefill's gate|up (32 rows, K 2048, 6 column tiles) and down
# (32 rows, K 352, 32 tiles), the same with 64 and 128 rows and with twice the columns, and
# three diagnostic shapes: shallower K, and the down shape with gate|up's 6 column tiles.
SHAPES = {
    "gu32": (2, 512, 6), "dn32": (2, 88, 32),
    "gu64": (4, 512, 6), "dn64": (4, 88, 32),
    "gu128": (8, 512, 6), "dn128": (8, 88, 32),
    "gu32w": (2, 512, 12), "gu128w": (8, 512, 12),
    "gu32_k128": (2, 128, 6), "gu32_k32": (2, 32, 6), "dn32_n6": (2, 88, 6),
}
MODES = {"i8": 0, "i4": 1, "i4b": 2}  # snax-dsv2-gemm-bench.c GB_I8, GB_I4, GB_I4B
OFFSETS = list(range(0, 256, 32))


def cases(name, b_offset):
    """[(shape, mode, B's bytes past a 256-byte boundary)] of a set, in run order."""
    if name == "unit":
        return [(s, m, b_offset) for s in ("gu32_k32", "dn32_n6") for m in ("i8", "i4", "i4b")]
    if name == "shapes":
        return [(s, m, b_offset) for s in SHAPES for m in ("i4b", "i4")]
    if name == "int8b":
        return [(s, m, b_offset) for s in ("gu32", "gu64", "gu32_k128")
                for m in ("i4b", "i4", "i8")]
    if name == "offsets":
        return [("gu32", m, o) for m in ("i4b", "i4", "i8") for o in OFFSETS]
    raise ValueError(f"no case set {name!r}: unit, shapes, int8b or offsets")


def b_blob(W, mode, mesh):
    if mode == "i8":
        return to_b(W, mesh)
    if mode == "i4":
        return pack_int4(to_b_pairs(W, mesh))
    return pack_int4(to_b(W, mesh))


def main():
    ap = argparse.ArgumentParser(description="Data for snax-dsv2-gemm-bench")
    ap.add_argument("--swcfg", type=pathlib.Path, required=True)
    ap.add_argument("--hwcfg", type=pathlib.Path, required=True)
    ap.add_argument("--header", type=pathlib.Path, required=True)
    ap.add_argument("--blob-dir", type=pathlib.Path, required=True)
    ap.add_argument("--set", default="unit", help="the cases: unit, shapes, int8b or offsets")
    args = ap.parse_args()
    prm = hjson.loads(args.swcfg.read_text())
    mesh = mesh_from_hwcfg(hjson.loads(args.hwcfg.read_text()))
    if mesh != (16, 4, 16):
        raise ValueError(f"the kernel's descriptors assume a (16, 4, 16) mesh, not {mesh}")
    mr, ku, nu = mesh
    k, b_offset = int(prm["k"]), int(prm["b_offset"])
    run = cases(args.set, b_offset)
    for s, m, o in run:
        if m != "i8" and SHAPES[s][2] % 2:
            raise ValueError(f"{s}: INT4 weights take column tiles in pairs")
        if o % 8 or not 0 <= o < 256:
            raise ValueError(f"B offset {o}: a multiple of 8 below 256")

    rng = np.random.default_rng(int(prm["seed"]))
    em = Emitter(args.blob_dir)
    em.c(f"// snax-dsv2-gemm-bench, set {args.set}: {len(run)} cases, seed {prm['seed']}.")
    em.define("GB_K", k, "the D port's RNE(acc * 2^-k)")
    em.define("GB_REPS", int(prm["reps"]), "runs of each case, the last measured")
    em.define("GB_NCASES", len(run))

    # one A and one weight per depth, sized for the largest shape of that depth in the set
    depths = sorted({SHAPES[s][1] for s, _, _ in run}, reverse=True)
    a_of, w_of, b_of = {}, {}, {}
    for K in depths:
        Mx = max(SHAPES[s][0] for s, _, _ in run if SHAPES[s][1] == K)
        Nx = max(SHAPES[s][2] for s, _, _ in run if SHAPES[s][1] == K)
        X = rng.integers(-128, 128, size=(mr * Mx, ku * K)).astype(np.int8)
        W = rng.integers(-8, 8, size=(ku * K, nu * Nx)).astype(np.int8)
        a_of[K] = (X, em.blob(f"gb_a{K}", to_a(X, mesh)))
        w_of[K] = W
        for mode in sorted({m for s, m, _ in run if SHAPES[s][1] == K}):
            b_of[K, mode] = em.blob(f"gb_b{K}_{mode}", b_blob(W, mode, mesh))

    y_of = {}
    for s in dict.fromkeys(s for s, _, _ in run):
        M, K, N = SHAPES[s]
        X, W = a_of[K][0], w_of[K]
        acc = X[:mr * M].astype(np.int64) @ W[:, :nu * N].astype(np.int64)
        y = d_port(acc, k)
        if np.any(np.isinf(y.astype(np.float32))):
            raise ValueError(f"{s}: an output overflows FP16 at k = {k}")
        y_of[s] = em.blob(f"gb_y_{s}", bits16(to_d(y, mesh)))

    rows, amax, bmax, dmax = [], 0, 0, 0
    for s, m, o in run:
        M, K, N = SHAPES[s]
        ab = M * K * mr * ku
        bb = N * K * ku * nu // (1 if m == "i8" else 2)
        amax, bmax, dmax = max(amax, ab), max(bmax, bb), max(dmax, M * N * mr * nu * 2)
        rows.append(f'    {{"{s}", {M}u, {K}u, {N}u, {MODES[m]}u, {o}u, {a_of[K][1]}, '
                    f'{b_of[K, m]}, {ab}u, {bb}u, {y_of[s]}}},')
    em.define("GB_ABYTES", amax, "the largest A a case reads")
    em.define("GB_BBYTES", bmax, "the largest B a case reads")
    em.define("GB_DBYTES", dmax, "the largest D a case writes, FP16")
    em.c("typedef struct {\n    const char *name;\n"
         "    uint32_t mt, kt, nt;  // tiles: 16 rows, 4 deep, 16 columns\n"
         "    uint32_t mode;        // GB_I8, GB_I4, GB_I4B\n"
         "    uint32_t boff;        // B's bytes past a 256-byte boundary; A sits on one\n"
         "    const int8_t *a, *b;\n    uint32_t abytes, bbytes;\n"
         "    const uint16_t *y;    // the expected D, FP16 in the D-layout\n} gb_case_t;")
    em.c("static const gb_case_t gb_cases[GB_NCASES] = {\n" + "\n".join(rows) + "\n};")
    em.write(args.header, args.blob_dir / "blobs.S")
    sys.stderr.write(f"[gemm-bench datagen] set {args.set}: {len(run)} cases, "
                     f"{em.bytes / 2**20:.2f} MiB of blobs\n")


if __name__ == "__main__":
    main()
