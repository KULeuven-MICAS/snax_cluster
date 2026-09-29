#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data for snax-dsv2-absorb: the absorbed per-head GEMVs of DeepSeek-V2-Lite's MLA, with the
# layer's own tensors from the golden pack (sw/apps/dsv2/util).
#
#   W_UK  q~_h = q_nope_h (1 x 128) . W_UK_h (128 x 512), k = 5   the query into the latent
#   W_UV  o_h  = o~_h     (1 x 512) . W_UV_h (512 x 128), k = 7   the latent output back out
#
# 16 heads each, 1 MiB of INT8 weights per projection, the latent norm's gain folded in.
# Per projection the app gets, all in DRAM:
#   <p>_a    the 16 heads' inputs x_h, 1 x K INT8 each, one after another
#   <p>_w    16 per-head weights in B-layout, K * N bytes each, consecutive
#   <p>_s    the per-column dequantisation factors, [16, N] FP16
#   <p>_y    expected RNE(x_h . W_h * 2^-k), [16, N] FP16
#   <p>_yd   expected y (.) s, [16, N] FP16

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))  # the util package
from util import golden  # noqa E402
from util.emit import Emitter  # noqa E402
from util.fp import bits16  # noqa E402
from util.layout import mesh_from_hwcfg  # noqa E402


def main():
    ap = argparse.ArgumentParser(description="Data for snax-dsv2-absorb")
    ap.add_argument("--swcfg", type=pathlib.Path, required=True)
    ap.add_argument("--hwcfg", type=pathlib.Path, required=True)
    ap.add_argument("--header", type=pathlib.Path, required=True)
    ap.add_argument("--blob-dir", type=pathlib.Path, required=True)
    ap.add_argument("--wbits", type=int, default=8, choices=(8, 4),
                    help="weight width: INT8, or INT4 through the B converter")
    args = ap.parse_args()
    prm = hjson.loads(args.swcfg.read_text())
    mesh = mesh_from_hwcfg(hjson.loads(args.hwcfg.read_text()))
    if mesh != (16, 4, 16):
        raise ValueError(f"the kernel's descriptors assume a (16, 4, 16) mesh, not {mesh}")
    hpt = int(prm["heads_per_task"])

    g = golden.make(seed=int(prm["seed"]), L=int(prm["L"]), wbits=args.wbits)
    H, P, d = g.hw, g.pack, g.dims
    if d.heads % hpt:
        raise ValueError(f"{d.heads} heads do not split into tasks of {hpt}")
    projs = [
        ("uk", "W_UK", H["qn8"], P.wuk, H["ks"]["uk"], H["qt_raw"], H["qt_s"], H["qt16"]),
        ("uv", "W_UV", H["ot8"], P.wuv, H["ks"]["uv"], H["oh_raw"], H["oh_s"], H["o16"]),
    ]

    em = Emitter(args.blob_dir)
    em.define("DSV2_DATA_WBITS", g.wbits, "the weights' width; the kernel's DSV2_WBITS must match")
    em.c(f"// DeepSeek-V2-Lite layer 1, golden seed {g.seed}, L = {g.L}.")
    em.define("ABS_HEADS", d.heads)
    em.define("ABS_HPT", hpt, "heads per GEMM task: the B buffer holds this many heads")
    em.define("ABS_NPROJ", len(projs))
    em.define("ABS_KMAX", max(p[3].q.shape[1] for p in projs))
    em.define("ABS_NMAX", max(p[3].q.shape[2] for p in projs))
    em.define("ABS_HEAD_BYTES", max(p[3].q.shape[1] * p[3].q.shape[2] for p in projs),
              "one head's weight, K * N")
    rows = []
    for key, name, x8, pk, k, y, s, yd in projs:
        Hh, K, N = pk.q.shape
        a = em.blob(f"{key}_a", np.asarray(x8, dtype=np.int8).reshape(-1))  # the 16 vectors x_h
        w = em.blob(f"{key}_w", pk.blob(mesh))
        sb = em.blob(f"{key}_s", bits16(s).reshape(-1))
        yb = em.blob(f"{key}_y", bits16(y).reshape(-1))
        ydb = em.blob(f"{key}_yd", bits16(yd).reshape(-1))
        rows.append(f'    {{"{name}", {a}, {w}, {sb}, {yb}, {ydb}, {K}u, {N}u, {k}u}},')
    em.c("typedef struct {\n    const char *name;\n    const int8_t *a, *w;\n"
         "    const uint16_t *s, *y, *yd;\n    uint32_t K, N, k;\n} abs_proj_t;")
    em.c("static const abs_proj_t abs_projs[ABS_NPROJ] = {\n" + "\n".join(rows) + "\n};")
    em.write(args.header, args.blob_dir / "blobs.S")
    sys.stderr.write(f"[dsv2-absorb datagen] {em.bytes / 2**20:.1f} MiB of blobs\n")


if __name__ == "__main__":
    main()
