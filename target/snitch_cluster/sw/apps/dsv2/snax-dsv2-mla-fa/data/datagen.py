#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data for snax-dsv2-mla-fa: DeepSeek-V2-Lite's multi-head latent attention for one token
# against L + 1 cached rows (the golden pack's L plus the token's own), from golden inputs:
#
#   D2   q~ (16 x 512) and the rotated q_pe (16 x 64), FP16 -> Q8, the [32 x 576] A operand
#        (rows 16..31 zero), two segment scales tied by s_q~ s_c = s_qpe s_kpe
#   A1   per key tile of Bc: S^T = K . Q^T (D port, k_s), the online softmax (V4), then
#        O^T += V^T . P^T with the column scaler; the last tile leaves as FP16 (k_o)
#   out  o~ = O16 (.) c / l per query, transposed, then quantised into W_UV's 16 A operands
#
# The cache is the two DRAM copies of layout.py (key_copy, value_copy).

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))  # the util package
from util import golden  # noqa E402
from util.emit import Emitter  # noqa E402
from util.fp import bits16, f32bits  # noqa E402
from util.hwmodel import P8_SCALE  # noqa E402
from util.layout import gemv_a_rep, key_copy, mesh_from_hwcfg, to_a, value_copy  # noqa E402


def p8_interleaved(p8, n_blk, ku, nu):
    """P8 [Bc keys, Br queries] as the interleaving quantiser writes it: B block (k, n) of
    P^T at (k * n_blk + n) * ku * nu, element (key, q) at (q % nu) * ku + key % ku."""
    bc, br = p8.shape
    out = np.zeros(bc * br, dtype=np.int8)
    for key in range(bc):
        for q in range(br):
            blk = (key // ku) * n_blk + q // nu
            out[blk * ku * nu + (q % nu) * ku + key % ku] = p8[key, q]
    return out


def main():
    ap = argparse.ArgumentParser(description="Data for snax-dsv2-mla-fa")
    ap.add_argument("--swcfg", type=pathlib.Path, required=True)
    ap.add_argument("--hwcfg", type=pathlib.Path, required=True)
    ap.add_argument("--header", type=pathlib.Path, required=True)
    ap.add_argument("--blob-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()
    prm = hjson.loads(args.swcfg.read_text())
    mesh = mesh_from_hwcfg(hjson.loads(args.hwcfg.read_text()))
    if mesh != (16, 4, 16):
        raise ValueError(f"the kernel's descriptors assume a (16, 4, 16) mesh, not {mesh}")
    bc, cap = int(prm["bc"]), int(prm["capacity"])

    g = golden.make(seed=int(prm["seed"]), L=int(prm["L"]), bc=bc)
    H, d, S = g.hw, g.dims, g.scales
    T = g.L + 1
    if T % bc or cap % bc or cap < T:
        raise ValueError(f"L + 1 = {T} keys must fill whole tiles of {bc} inside capacity {cap}")
    att = H["att"]
    rows = np.concatenate([np.concatenate([g.c8, g.kpe8], axis=1),
                           np.concatenate([H["c8_new"], H["kpe8_new"]])[None]])

    em = Emitter(args.blob_dir)
    em.c(f"// DeepSeek-V2-Lite layer 1 MLA, golden seed {g.seed}: {T} keys in tiles of {bc}.")
    em.define("BC", bc, "keys per tile")
    em.define("NT", T // bc, "key tiles")
    em.define("CAP", cap, "tokens the cache copies hold")
    em.define("K_S", H["ks"]["s"], "D-port shift of the scores, depth 576")
    em.define("K_O", H["k_o"], f"D-port shift of O, depth {T}")
    em.define("A_EXP", f"0x{f32bits(H['a_exp']):08X}u", f"exp scale a' = {H['a_exp']:.6g}")
    em.define("A_N", f"0x{f32bits(H['a_n']):08X}u", "RSQRT scale 1/c, c = 2^k_o s_c / 127")
    em.define("P8_INV", f"0x{f32bits(P8_SCALE):08X}u", "P8 = rne(127 P)")
    em.define("INV_QT", f"0x{f32bits(S.qt.inv):08X}u", "quantise q~")
    em.define("INV_QPE", f"0x{f32bits(S.qpe.inv):08X}u", "quantise the rotated q_pe")
    em.define("INV_OT", f"0x{f32bits(S.ot.inv):08X}u", "quantise o~ for W_UV")
    em.blob("key", key_copy(rows, cap, mesh))
    em.blob("val", value_copy(rows[:, :d.kv_rank], cap, mesh))
    em.blob("qt16", bits16(H["qt16"]).reshape(-1))
    em.blob("qpe16", bits16(H["qpe_rot"]).reshape(-1))
    em.blob("g_q8", to_a(H["Q8"], mesh))
    em.blob("g_s", bits16(np.concatenate([t["S16"] for t in att["tiles"]])).reshape(-1))
    em.blob("g_m", bits16(att["m"]))
    em.blob("g_l", bits16(att["l"]))
    em.blob("g_p8", p8_interleaved(att["tiles"][-1]["p8"].astype(np.int8), 2, 4, 16))
    em.blob("g_o16", bits16(H["O16"]).reshape(-1))
    em.blob("g_rsc", bits16(H["rsc16"]))
    em.blob("g_lsq", bits16(H["att"]["l"]))
    em.blob("g_ot", bits16(H["ot16"]).reshape(-1))
    em.blob("g_auv", np.concatenate([gemv_a_rep(H["ot8"][h], mesh) for h in range(d.heads)]))
    em.write(args.header, args.blob_dir / "blobs.S")
    moved = sum(int((t["m"] > m0).sum()) for t, m0 in
                zip(att["tiles"][1:], [t["m"] for t in att["tiles"][:-1]]))
    sys.stderr.write(f"[dsv2-mla-fa datagen] {em.bytes / 2**10:.0f} KiB of blobs; the running max "
                     f"moved {moved} times after tile 0; l in [{float(att['l'][:16].min()):.1f}, "
                     f"{float(att['l'][:16].max()):.1f}]\n")


if __name__ == "__main__":
    main()
