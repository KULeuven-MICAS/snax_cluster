#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data for snax-dsv2-simd: the SIMD stages of DeepSeek-V2-Lite layer 1 on the layer's own
# tensors (the golden pack, sw/apps/dsv2/dsv2), each from its golden input:
#
#   V1+V2  input norm       x (2048)          -> xn (FP16) -> the W_Q/W_DKV A operand (INT8)
#   V1+V2  latent norm      c = ckv[:512]     -> cn (FP16) -> the cache row's c (INT8)
#   V3     RoPE at L        q_pe (16 x 64), k_pe (64) as one 1,088-wide row
#   V1+V2  post-attn norm   h (2048)          -> hn (FP16) -> the router/expert A operand
#   V6+V2  SwiGLU           an expert's [gate | up] (2 x 1408), the shared (2 x 2816)
#                           -> silu(gate) (.) up (FP16) -> the down GEMV's A operand
#   V1+V2  unit rows        a 2048- and a 512-wide row whose mean square is EXACTLY 1: the
#                           1/rms is exactly 1, so norm and quantise must be bit-exact
#
# The A operands' goldens are their INT8 values: the app compares the row the GEMV reads.

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from dsv2 import golden, hwmodel  # noqa E402
from dsv2.emit import Emitter  # noqa E402
from dsv2.fp import F16, bits16, f32bits, quant_i8  # noqa E402
from dsv2.layout import mesh_from_hwcfg  # noqa E402


def unit_row(n, rng):
    """n values with sum(x^2) = n exactly: 5/16 of them +-0.5, 3/16 +-1.5, half +-1."""
    b = 3 * n // 16
    a = 5 * n // 16
    v = np.array([0.5] * a + [1.5] * b + [1.0] * (n - a - b), dtype=np.float64)
    assert (v * v).sum() == n
    v *= rng.choice([-1.0, 1.0], size=n)
    return rng.permutation(v).astype(F16)


def main():
    ap = argparse.ArgumentParser(description="Data for snax-dsv2-simd")
    ap.add_argument("--swcfg", type=pathlib.Path, required=True)
    ap.add_argument("--hwcfg", type=pathlib.Path, required=True)
    ap.add_argument("--header", type=pathlib.Path, required=True)
    ap.add_argument("--blob-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()
    prm = hjson.loads(args.swcfg.read_text())
    mesh = mesh_from_hwcfg(hjson.loads(args.hwcfg.read_text()))
    if mesh != (16, 4, 16):
        raise ValueError(f"the A operand's row write assumes a (16, 4, 16) mesh, not {mesh}")

    g = golden.make(seed=int(prm["seed"]), L=int(prm["L"]))
    H, d, S = g.hw, g.dims, g.scales
    rng = np.random.default_rng([int(prm["seed"]), 7])
    em = Emitter(args.blob_dir)
    em.c(f"// DeepSeek-V2-Lite layer 1, golden seed {g.seed}, token at position {g.pos}.")

    # ---- the norms and their quantisers ---------------------------------------------------
    em.define("INV_X", f"0x{f32bits(S.x.inv):08X}u", "quantise the normed input")
    em.define("INV_C", f"0x{f32bits(S.c.inv):08X}u", "quantise the normed latent (cache)")
    em.define("INV_H", f"0x{f32bits(S.h.inv):08X}u", "quantise the normed post-attention state")
    em.blob("x16", bits16(H["x16"]))
    em.blob("g_xn", bits16(H["xn"]))
    em.blob("g_xa", np.asarray(H["xq"], dtype=np.int8))
    em.blob("ckv16", bits16(H["ckv16"]))
    em.blob("g_cn", bits16(H["cn16"]))
    em.blob("g_c8", H["c8_new"])
    em.blob("h16", bits16(H["h16"]))
    em.blob("g_hn", bits16(H["hn"]))
    em.blob("g_ha", np.asarray(H["hq"], dtype=np.int8))

    # ---- RoPE: the 16 heads' q_pe and the one k_pe, as one row, with per-pair tables -------
    em.define("ROPE_HEADS", d.heads)
    em.define("ROPE_DIM", d.q_rope)
    em.define("Q_HEAD", d.q_head, "a head's slice of q: [q_nope | q_pe]")
    em.define("Q_NOPE", d.q_nope)
    em.define("KV_RANK", d.kv_rank)
    em.blob("q16", bits16(H["q16"]))
    c_rep = np.repeat(H["cos16"], 2)
    s_sgn = np.empty(d.q_rope, dtype=F16)
    s_sgn[0::2] = -H["sin16"]
    s_sgn[1::2] = H["sin16"]
    em.blob("rope_cos", bits16(np.tile(c_rep, d.heads + 1)))
    em.blob("rope_sin", bits16(np.tile(s_sgn, d.heads + 1)))
    em.blob("g_rope", bits16(np.concatenate([H["qpe_rot"].reshape(-1), H["kpe_rot"]])))

    # ---- SwiGLU: the first slot's expert and the shared experts ----------------------------
    em.define("I_EXPERT", d.moe_inter)
    em.define("I_SHARED", d.shared_inter)
    s0, sh = H["slots"][0], H["shared"]
    e0 = int(H["ids"][0])
    em.define("INV_AE", f"0x{f32bits(S.expert_a[e0].inv):08X}u", f"expert {e0}'s activation")
    em.define("INV_AS", f"0x{f32bits(S.shared_a.inv):08X}u", "the shared experts' activation")
    for key, sl in (("e", s0), ("s", sh)):
        em.blob(f"{key}_g16", bits16(sl["g16"]))
        em.blob(f"g_{key}_sg", bits16(sl["sg16"]))
        em.blob(f"g_{key}_a16", bits16(sl["a16"]))
        em.blob(f"g_{key}_a8", np.asarray(sl["a8"], dtype=np.int8))

    # ---- the unit rows: mean square exactly 1 --------------------------------------------
    for n in (2048, 512):
        u = unit_row(n, rng)
        un = hwmodel.rmsnorm(u)
        assert np.array_equal(bits16(un), bits16(u)), "1/rms of a unit row must be exactly 1"
        em.blob(f"unit{n}", bits16(u))
        em.blob(f"g_unit{n}a", np.asarray(quant_i8(un, S.x.inv), dtype=np.int8))
    em.write(args.header, args.blob_dir / "blobs.S")
    sys.stderr.write(f"[dsv2-simd datagen] {em.bytes / 2**10:.0f} KiB of blobs\n")


if __name__ == "__main__":
    main()
