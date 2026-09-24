#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data for snax-llm-qkv-to-fa: RMSNorm -> quantise -> Q/K/V projections -> FlashAttention
# operands, the attention front half of HeMAiA's llm_layer_4cluster.
#
# THE SAME TENSORS AS THE LAYER. x and the three weights are drawn from
# np.random.default_rng(7) in exactly the order util/sim/llm/llm_layer_data.py draws them,
# so norm1, proj_q and the rest are the layer's own values, not look-alikes.
#
# WHAT IS STAGED, AND IN WHICH LAYOUT. Weights are offline data, so each is staged in the
# layout its GEMM reads, at no run-time cost:
#   wq_b, wk_b, wv_b   B-layout of W [d_in, d_out]     x.W, the layer's own projections
#   wvt_a              A-layout of Wv^T [d_out, d_in]  V^T = Wv^T . Xn^T, the swapped form
# The swapped V projection is what makes V^T -- the operand FlashAttention's P.V matmul
# reads -- come out of the GEMM with no transpose. Its B operand is Xn^T in B-layout, and
# on a mesh with meshRow == meshCol that is byte-for-byte the A-layout of Xn the other two
# projections already read. See the .c for the derivation.
#
# THE GOLDENS model the device, not the maths: the reduce accumulates in FP32 and narrows
# the scalar to FP16, the rsqrt is FP16, every stage narrows to FP16, the quantiser is
# symmetric (+-127) with ties-to-even, and the D port narrows INT32 to FP16 with RNE. The
# one thing a golden cannot pin exactly is the rsqrt ROM, which is within 1 FP16 ULP of
# the true value; the app checks with tolerances there and bit-exactly everywhere else.

import argparse
import math
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
from data_utils import format_vector_definition  # noqa E402
from snax_utils import int32_to_fp16_golden  # noqa E402

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from blocked_nest import blocked_nest  # noqa E402

# Meshes whose operand layouts the app writes with HeMAiA's descriptor-driven blocked pass.
# The SIMD pass is mesh-agnostic hardware -- it never talks to the array -- so these run
# on this cluster whatever GEMM it has: the real cfgs' shapes, and a few more.
BLOCKED_MESHES = [(16, 4, 16), (16, 8, 16), (16, 8, 8), (1, 32, 32), (1, 16, 32),
                  (8, 8, 8), (32, 4, 32), (16, 32, 16)]

FP16_MAX = 65504.0
# The layer's weight quantisation scale (llm_layer_data.W_SCALE): the dequantise factor
# after a projection is 1 / (scale_n1 * W_SCALE).
W_SCALE = 16.0


def _mesh(hw):
    """(meshRow, tileSize, meshCol) = VersaCore's (Mu, Ku, Nu). One shape, one dtype."""
    acc = hw["snax_versacore_core_template"]["snax_acc_cfg"][0]
    unrolling = acc["snax_versacore_spatial_unrolling"]
    assert len(unrolling) == 1 and len(unrolling[0]) == 1, \
        "this kernel programs array_shape = data_type = 0; the cfg declares more"
    return tuple(unrolling[0][0])


# ---- the array's layouts, as block_gemm_golden_model and the streamer read them ----------
def to_a(R, mr, ts):
    """A[m][k][r][s] = R[m*mr + r, k*ts + s]."""
    M, K = R.shape
    return R.reshape(M // mr, mr, K // ts, ts).transpose(0, 2, 1, 3).reshape(-1)


def to_b(R, ts, mc):
    """B[n][k][c][s] = R[k*ts + s, n*mc + c]  (R is [K, N])."""
    K, N = R.shape
    return R.reshape(K // ts, ts, N // mc, mc).transpose(2, 0, 3, 1).reshape(-1)


def to_d(R, mr, mc):
    """D[m][n][r][c] = R[m*mr + r, n*mc + c]."""
    M, N = R.shape
    return R.reshape(M // mr, mr, N // mc, mc).transpose(0, 2, 1, 3).reshape(-1)


# ---- the device's arithmetic -------------------------------------------------------------
def int8_scale_for(peak):
    """llm_layer_data.int8_scale_for: the largest power of two keeping `peak` in int8."""
    if peak <= 0:
        return 1.0
    return float(2.0 ** math.floor(math.log2(127.0 / peak)))


def quant(x16, scale):
    """Fp16ToInt8: FP32 product, clamp, round ties-to-even, SYMMETRIC saturate to +-127."""
    v = x16.astype(np.float32) * np.float32(scale)
    v = np.clip(v, -128.0, 128.0)
    return np.clip(np.rint(v), -127, 127).astype(np.int8)


def d_port_fp16(acc):
    """The D port's Int32ToFp16Converter, bit for bit."""
    flat = np.asarray(acc, dtype=np.int64).reshape(-1)
    bits = np.array([int32_to_fp16_golden(int(v)) for v in flat], dtype=np.uint16)
    return bits.view(np.float16).reshape(np.shape(acc))


def f32bits(x):
    return int(np.asarray(x, dtype=np.float32).view(np.uint32))


def emit(**kw):
    T, d = int(kw["TOKENS"]), int(kw["D_MODEL"])
    mr, ts, mc = _mesh(kw)
    if T != 32:
        raise ValueError(f"TOKENS must be 32 (FlashAttention's Br and Bc), got {T}")
    if d % 32 or d & (d - 1):
        raise ValueError(f"D_MODEL must be a power of two and a multiple of 32, got {d}")
    if mr != mc:
        raise ValueError(
            f"meshRow ({mr}) != meshCol ({mc}): B-layout of Q^T is then NOT the A-layout "
            f"of Q, and neither Q nor the swapped V projection can reuse the A operand.")
    if mr % 8 or T % mr or d % mr or d % ts:
        raise ValueError(f"shape [{T}, {d}] does not tile the ({mr}, {ts}, {mc}) array")

    # ---- the layer's draws, in the layer's order ----------------------------------------
    rng = np.random.default_rng(seed=7)
    x = (rng.integers(-4, 4, size=(T, d)).astype(np.float32) / 4.0).astype(np.float16)
    w = {nm: rng.integers(-2, 2, size=(d, d), dtype=np.int8) for nm in ("q", "k", "v")}

    # ---- RMSNorm, as the SIMD computes it ------------------------------------------------
    xf = x.astype(np.float32)
    ssq = np.array([np.float16((xf[r] ** 2).sum(dtype=np.float32)) for r in range(T)],
                   dtype=np.float16)
    mean = ssq.astype(np.float32) / np.float32(d)                  # exact: d = 2^k
    inv = (np.float32(1.0) / np.sqrt(mean)).astype(np.float16)
    n1 = (xf * inv.astype(np.float32)[:, None]).astype(np.float16)
    if float(ssq.astype(np.float32).max()) >= FP16_MAX:
        raise ValueError("a row's sum of squares overflows the FP16 reduce output")

    s_n1 = int8_scale_for(float(np.abs(n1.astype(np.float32)).max()))
    n1q = quant(n1, s_n1)

    # ---- the projections: D port first, THEN the dequantise (the layer's order) -----------
    dq = 1.0 / (s_n1 * W_SCALE)
    y16 = {nm: d_port_fp16(n1q.astype(np.int32) @ w[nm].astype(np.int32)) for nm in w}
    yd = {nm: (y16[nm].astype(np.float32) * np.float32(dq)).astype(np.float16) for nm in w}

    # ---- FlashAttention's operands --------------------------------------------------------
    # K and V keep the layer's shared scale. Q DOES NOT, and that is a finding, not a
    # tuning choice: FlashAttention narrows S = K.Q^T from INT32 to FP16 on the D port with
    # no scale in between, so |S| must stay under 65504. At the layer's s_qk this data
    # reaches ~2.9e5 -- every overflowing score becomes inf, and exp(inf - inf) is NaN.
    # Real attention divides S by sqrt(d) anyway; putting a power of two of that into Q's
    # quantisation step is where it costs nothing. So Q's scale is the largest power of two
    # at or below s_qk that keeps THIS data's worst score in FP16, derived like every other
    # scale here rather than fixed.
    peak = max(float(np.abs(yd[nm].astype(np.float32)).max()) for nm in w)
    s_kv = int8_scale_for(peak)
    k8, v8 = quant(yd["k"], s_kv), quant(yd["v"], s_kv)
    s_q = s_kv
    while True:
        q8 = quant(yd["q"], s_q)
        s = k8.astype(np.int64) @ q8.astype(np.int64).T              # S^T: [key, query]
        if np.abs(s).max() < FP16_MAX:
            break
        s_q /= 2.0
    s_at_skv = int(np.abs(k8.astype(np.int64) @ quant(yd["q"], s_kv).astype(np.int64).T).max())
    s16 = d_port_fp16(s)                                             # [key][query]

    # The P.V check. P here is NOT a softmax -- it is a small non-negative INT8 operand
    # chosen so O^T = V^T.P^T stays inside FP16 and can come back through the converter
    # and be checked element for element. FlashAttention accumulates O in INT32 instead;
    # what this matmul proves is that V^T is laid out the way that matmul reads it.
    prng = np.random.default_rng(seed=0x5A)
    pt = prng.integers(0, 4, size=(T, T), dtype=np.int8)             # P^T: [key, query]
    o = v8.astype(np.int64).T @ pt.astype(np.int64)                  # O^T: [d, query]
    if np.abs(o).max() >= FP16_MAX:
        raise ValueError("the P.V check overflows FP16; shrink P")
    o16 = d_port_fp16(o)

    # ---- blocked-pass descriptors, one per (mesh, operand, precision) that has a nest ----
    pitch_a = (d // 32 + 1) * 64 + 8          # the TAP reduce's pitch
    pitch_b = 2 * T + 8                        # y^T from the sticky multiply
    cases = []
    for mu, ku, nu in BLOCKED_MESHES:
        for kind, (R, C, P, mb, mul) in enumerate(((T, d, pitch_a, mu, True),
                                                  (d, T, pitch_b, nu, False))):
            for ob in (2, 1):
                try:
                    n = blocked_nest(R, C, P, mb, ku, ob, fused_mul=mul)
                except ValueError:
                    continue
                rd = list(n.rd) + [(1, 0)] * (3 - len(n.rd))
                wr = list(n.wr) + [(1, 0)] * (3 - len(n.wr))
                cases.append([kind, 1 if ob == 1 else 0, mu, ku, nu, mb, n.pitch, n.rd_lane,
                              *[x for b, s in rd for x in (b, s)], n.wr_lane,
                              *[x for b, s in wr for x in (b, s)], n.reps, n.rep_rd,
                              n.rep_wr])

    u16 = lambda a: np.asarray(a, dtype=np.float16).view(np.uint16).reshape(-1)  # noqa E731
    i8 = lambda a: np.asarray(a, dtype=np.int8).reshape(-1)                      # noqa E731

    out = ["#include <stdint.h>"]
    # Constants as MACROS, not globals: a non-const data.h global lives in DRAM here, and
    # every read of one inside a CSR sequence is an L3 round trip.
    defs = [
        ("T_TOK", T, "tokens = FlashAttention's Br and Bc"),
        ("D_MODEL", d, "d_model = d_head"),
        ("LOG2D", d.bit_length() - 1, "for the exact 1/D"),
        ("MESH_ROW", mr, "VersaCore Mu"),
        ("TILE_SIZE", ts, "VersaCore Ku"),
        ("MESH_COL", mc, "VersaCore Nu"),
        ("SCALE_N1_BITS", f"0x{f32bits(s_n1):08X}u", f"quantise norm1, x{s_n1:g}"),
        ("DQ_PROJ_BITS", f"0x{f32bits(dq):08X}u", f"dequantise a projection, x{dq:g}"),
        ("SCALE_Q_BITS", f"0x{f32bits(s_q):08X}u", f"quantise Q, x{s_q:g}"),
        ("SCALE_KV_BITS", f"0x{f32bits(s_kv):08X}u", f"quantise K and V, x{s_kv:g}"),
    ]
    out += ["\n".join(f"#define {n:<14} {v:<12} // {c}" for n, v, c in defs)]
    out += [f"// |K.Q^T| at the layer's shared scale x{s_kv:g}: {s_at_skv} (FP16 max 65504);\n"
            f"// at Q's scale x{s_q:g}: {int(np.abs(s).max())}. |O^T| max {int(np.abs(o).max())}."]

    def vec(ctype, name, arr):
        hexb = {"uint16_t": 16}.get(ctype)
        return format_vector_definition(ctype, name, arr, alignment=64,
                                        hex_bits=hexb, cast_hex=hexb is not None)

    out += ["// kind 0 = A of y (row_major, fused multiply), 1 = B of y (from y^T).\n"
            "typedef struct {\n  uint32_t kind, out_i8, mu, ku, nu, mb, pitch, rd_lane;\n"
            "  uint32_t rd[6];   // (bound, stride) x 3, innermost first\n"
            "  uint32_t wr_lane;\n  uint32_t wr[6];\n  uint32_t reps, rep_rd, rep_wr;\n"
            "} blk_case_t;\n"
            f"#define N_BLK_CASES {len(cases)}\n"
            "static const blk_case_t blk_cases[N_BLK_CASES] = {\n" +
            ",\n".join("  {" + ", ".join(str(v) for v in c) + "}" for c in cases) + "\n};"]
    out += [vec("uint16_t", "x_in", u16(x))]
    out += [vec("int8_t", "wq_b", i8(to_b(w["q"], ts, mc)))]
    out += [vec("int8_t", "wk_b", i8(to_b(w["k"], ts, mc)))]
    out += [vec("int8_t", "wv_b", i8(to_b(w["v"], ts, mc)))]
    out += [vec("int8_t", "wvt_a", i8(to_a(np.ascontiguousarray(w["v"].T), mr, ts)))]
    out += [vec("int8_t", "pt_b", i8(to_b(pt, ts, mc)))]
    # goldens
    out += [vec("uint16_t", "g_n1", u16(n1))]                        # packed [T, d]
    out += [vec("int8_t", "g_n1q_a", i8(to_a(n1q, mr, ts)))]         # A/i8
    out += [vec("uint16_t", "g_projq_d", u16(to_d(yd["q"], mr, mc)))]  # HeMAiA's proj_q check
    out += [vec("int8_t", "g_q8", i8(to_a(q8, mr, ts)))]             # A(Q) == B(Q^T)
    out += [vec("int8_t", "g_k8", i8(to_a(k8, mr, ts)))]             # A(K)
    out += [vec("int8_t", "g_vt8", i8(to_a(np.ascontiguousarray(v8.T), mr, ts)))]  # A(V^T)
    out += [vec("uint16_t", "g_s16", u16(s16))]                      # [key][query]
    out += [vec("uint16_t", "g_o16", u16(o16))]                      # O^T [d][query]
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Data for snax-llm-qkv-to-fa")
    ap.add_argument("--swcfg", type=pathlib.Path, required=True)
    ap.add_argument("--hwcfg", type=pathlib.Path, required=True)
    args = ap.parse_args()
    with args.swcfg.open() as f:
        param = hjson.loads(f.read())
    with args.hwcfg.open() as f:
        hw = hjson.loads(f.read())
    print(emit(**{**param, **hw}))


if __name__ == "__main__":
    main()
