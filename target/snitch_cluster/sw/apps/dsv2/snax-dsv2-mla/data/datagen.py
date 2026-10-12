#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data for snax-dsv2-mla: DeepSeek-V2-Lite layer 1's MLA block for one token, x -> x + MLA(x)
# (stages 1 to 13), from the golden pack (sw/apps/dsv2/util). The block kernel's inputs -- the
# weights, their factors, the cache holding the L cached rows, the pass -- come from
# dsv2.appdata; this app adds the goldens.
#
# GOLDENS are the device model's chained values (hwmodel.run), which reproduce the SIMD's
# tables, fused multiply-adds and orders of summation bit for bit, so every stage is checked
# exactly: xn, ckv, cn, the cache row, q, the rotated q_pe | k_pe, Q8, the last two
# tiles' scores, m, l, the last P8, c / l, o~, the heads' outputs, W_O's A operand, its raw
# output, attn and h = x + attn, and both cache copies after the append. The per-query ones
# (scores, m, l, P8, c / l) are in the kernel's score-plane layouts (util/layout.py planes,
# p8_halves): one token, so one plane of its 16 heads.
#
# TEST SIZE (params.hjson `dims`, L): a narrower hidden state and a short cache, so the block
# streams a few MiB; the attention's dimensions are DeepSeek-V2-Lite's.

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))  # the util package
from util import appdata, golden, simd  # noqa E402
from util.emit import Emitter  # noqa E402
from util.fp import bits16  # noqa E402
from util.layout import key_copy, mesh_from_hwcfg, p8_halves, planes, to_a, value_copy  # noqa E402


def main():
    ap = argparse.ArgumentParser(description="Data for snax-dsv2-mla")
    ap.add_argument("--swcfg", type=pathlib.Path, required=True)
    ap.add_argument("--hwcfg", type=pathlib.Path, required=True)
    ap.add_argument("--header", type=pathlib.Path, required=True)
    ap.add_argument("--blob-dir", type=pathlib.Path, required=True)
    ap.add_argument("--wbits", type=int, default=8, choices=(8, 4),
                    help="weight width: INT8, or INT4 through the B converter")
    args = ap.parse_args()
    prm = hjson.loads(args.swcfg.read_text())
    hw = hjson.loads(args.hwcfg.read_text())
    mesh = mesh_from_hwcfg(hw)
    simd.use_hwcfg(hw)  # the SIMD core the goldens model
    if mesh != (16, 4, 16):
        raise ValueError(f"the kernel's descriptors assume a (16, 4, 16) mesh, not {mesh}")
    bc, cap = int(prm["bc"]), int(prm["capacity"])

    g = golden.make(seed=int(prm["seed"]), L=int(prm["L"]), bc=bc, wbits=args.wbits,
                    dims=prm.get("dims"))
    H, d = g.hw, g.dims
    att = H["att"]
    T = g.L + 1
    if cap % bc or cap < T:
        raise ValueError(f"capacity {cap} must be whole tiles of {bc} and hold {T} tokens")

    em = Emitter(args.blob_dir)
    em.define("DSV2_DATA_WBITS", g.wbits, "the weights' width; the kernel's DSV2_WBITS must match")
    em.c(f"// DeepSeek-V2-Lite layer 1 MLA block, golden seed {g.seed}: the token at position "
         f"{g.pos}, {T} keys in tiles of {bc}.")
    appdata.model_defines(em, g, bc, cap)
    appdata.mla_weights(em, g, mesh)
    appdata.cache(em, g, cap, mesh)
    appdata.mla_passes(em, [appdata.golden_pass(g)], d.heads)

    # ---- goldens, in stage order ------------------------------------------------------------
    new_row = np.concatenate([H["c8_new"], H["kpe8_new"]])
    rows = np.concatenate([appdata.cache_rows(g), new_row[None]])
    em.blob("g_xn", bits16(H["xn"]))
    em.blob("g_ckv", bits16(H["ckv16"]))
    em.blob("g_cn", bits16(H["cn16"]))
    em.blob("g_row", new_row)
    em.blob("g_q", bits16(H["q16"]))
    em.blob("g_rot", bits16(np.concatenate([H["qpe_rot"].reshape(-1), H["kpe_rot"]])))
    br = d.heads  # one token: one score plane, Br = its 16 heads
    em.blob("g_q8", to_a(H["Q8"][:br], mesh))
    em.blob("g_s", bits16(np.stack([planes(t["S16"][:, :br]) for t in att["tiles"][-2:]])))
    em.blob("g_m", bits16(att["m"][:br]))
    em.blob("g_l", bits16(att["l"][:br]))
    em.blob("g_p8", p8_halves(att["tiles"][-1]["p8"][:, :br], mesh))
    em.blob("g_rsc", bits16(H["rsc16"][:br]))
    em.blob("g_ot", bits16(H["ot16"]).reshape(-1))
    em.blob("g_oh", bits16(H["o16"]).reshape(-1))
    em.blob("g_oa", np.asarray(H["o8"], dtype=np.int8))  # the A operand's row 0
    em.blob("g_ya", bits16(H["a_raw"]))
    em.blob("g_attn", bits16(H["attn16"]))
    em.blob("g_h", bits16(H["h16"]))
    em.blob("g_key", key_copy(rows, cap, mesh))
    em.blob("g_val", value_copy(rows[:, :d.kv_rank], cap, mesh))
    em.write(args.header, args.blob_dir / "blobs.S")
    sys.stderr.write(f"[dsv2-mla datagen] {em.bytes / 2**20:.1f} MiB of blobs; l in "
                     f"[{float(att['l'][:d.heads].min()):.1f}, {float(att['l'][:d.heads].max()):.1f}]\n")


if __name__ == "__main__":
    main()
