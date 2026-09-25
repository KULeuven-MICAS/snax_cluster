#!/usr/bin/env python3
# Data and streamer descriptors for FlashAttention on VersaCore.
#
# Emits the operands, the streamer descriptors for both matmul shapes, the tile geometry
# as compile-time constants, and a float model of the softmax for the kernel to check
# against. The block order reaching TCDM is (M, N, meshRow, meshCol), which is what
# block_gemm_golden_model() assumes.

# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

import numpy as np
import argparse
import re
import pathlib
import hjson
import sys
import os
import math

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402

# Add golden model path
from snax_utils import (  # noqa E402
    block_gemm_golden_model,
    align_wide_addr,
)  # noqa E402

np.random.seed(42)


def _acc(kwargs):
    return kwargs["snax_versacore_core_template"]["snax_acc_cfg"][0]


def _mesh(kwargs):
    """meshRow, tileSize, meshCol -- VersaCore's (Mu, Ku, Nu) spatial unrolling.

    VersaCore carries one triple per [data type][array shape] and selects between them at
    run time with two CSRs. This cluster declares exactly one of each, so both CSRs are
    always 0 and the mesh is fixed. The assert is what makes that safe to assume: adding a
    shape to the cfg fails the generator rather than silently emitting descriptors for the
    wrong one.
    """
    unrolling = _acc(kwargs)["snax_versacore_spatial_unrolling"]
    assert len(unrolling) == 1 and len(unrolling[0]) == 1, (
        "this kernel emits array_shape = data_type = 0; the cfg declares "
        "%d data type(s) and %d array shape(s)" % (len(unrolling), len(unrolling[0]))
    )
    return tuple(unrolling[0][0])


def score_scale_split(a):
    """Split the softmax temperature a into a power of two for the converter and a remainder for exp.

    The GEMM's scores are exact INT32 and with full-range INT8 reach 127^2 * d (2,064,512 at
    d = 128), 31x past FP16's 65,504. The D-port converter therefore applies a power-of-two
    scale while it rounds, S16 = RNE(S * 2^-k), which moves only the exponent: no precision is
    lost, and the scores land where the softmax works. The rest of a is applied by StreamMap in
    the exp, which multiplies anyway:

        a * S  =  a' * (S * 2^-k)     with  a' = a * 2^k

    k is capped at 14 (the converter's range: 2^-14 is FP16's smallest normal number, so no
    integer can underflow) and at the smallest shift that keeps the worst-case score finite.
    """
    k = int(np.clip(np.floor(-np.log2(a)), 0, 14))
    return k, float(a * 2.0 ** k)


def _scores(Kj, B, M, N, K, meshRow, tileSize, meshCol):
    """S^T of one KV tile as [key][query] -- how the D32 port lays the FP16 tile down."""
    D = block_gemm_golden_model(
        M, K, N, meshRow, tileSize, meshCol, Kj, B, 0, 0,
        np.zeros(M * N * meshRow * meshCol, dtype=np.int64),
    )
    S = np.asarray(D, dtype=np.int64).reshape(M, N, meshRow, meshCol)
    return S.transpose(0, 2, 1, 3).reshape(M * meshRow, N * meshCol)


def p8_interleave_offset(key, q, N, tileSize, meshCol):
    """Byte offset of P8(key, query) in the buffer the INTERLEAVE quantiser writes.

    Input beat b of 4-beat group g is key 4g+b; output beat (2g + q/16) holds, at byte
    4*(q%16)+b, lane q of that input beat. Output beat h of group g is therefore B block
    (k = g, n = h), k-major: block (k, n) at (k*N + n)*64. The kernel's P8_OFF() places the
    same blocks P8_SLOT bytes apart instead of 64.
    """
    blk = (key // tileSize) * N + q // meshCol
    return blk * tileSize * meshCol + (q % meshCol) * tileSize + key % tileSize


def emit_attention_golden(Ks, Vs, B, **kwargs):
    """A float model of the WHOLE online softmax over NKV distinct KV tiles.

    Every KV tile has its own K and V, so the running maximum moves and corr is not 1.
    The model follows the hardware's sequence at the precision each stage works in:

        S16    exact INT32 out of the mesh, RNE to FP16 on the D32 port
        m      max over keys of S16, running across tiles           (exact)
        corr   exp(a * fp16(m_old - m_new))                        EW0 ADD -> Map EXP
        P      exp(a * fp16(S16 - m_new)), stored FP16 in-chain     EW0 ADD -> Map EXP
        rowsum sum over keys of P in FP32, narrowed to FP16         Reduce TAP
        P8     sat(rne(127 * P))                                    Fp16ToInt8, inv_scale 127
        l      fp16(rowsum + fp16(corr * l_old))                    EW1 MUL, Reduce ADD
        O      rne(O * corr[col]) + V^T . P8^T                      Int32ColumnScale on C, PV

    O is built from the ATTENTION MATH -- P8 as a matrix [key][query], handed to the block
    model as the canonical B operand P^T -- never from the bytes the quantiser writes: a
    golden that reinterpreted P8's memory as B would reproduce a layout error in the
    hardware instead of catching it. Whether the interleaved bytes really are that operand
    is asserted separately below.

    The only thing not reproduced bit for bit is the SIMD exponential (a LUT, ~1 ULP), so
    P8 can land one step off numpy's, and corr one ULP off. O is therefore checked
    against a per-element tolerance scaled by sum|terms| (the standard bound for a sum that
    cancels); m is exact, and rowsum, l and the sampled P8 are checked in ULPs / exactly
    away from rounding boundaries.
    """
    meshRow, tileSize, meshCol = _mesh(kwargs)
    M, N, K, NKV = kwargs["M"], kwargs["N"], kwargs["K"], int(kwargs["NKV"])
    Bc, Br = M * meshRow, N * meshCol
    f16, f32 = np.float16, np.float32
    shift, a_exp = score_scale_split(float(kwargs["SCORE_SCALE"]))
    a32 = f32(a_exp)          # the part of a the exp applies; 2^-shift is in the converter
    q127 = f32(127.0)
    s2_m = (K * tileSize) // meshRow      # d / meshRow
    s2_k = Bc // tileSize                 # Bc / tileSize
    assert Bc % tileSize == 0, "the interleave groups keys by tileSize = 4"

    m16 = np.full(Br, -65504.0, dtype=f16)   # the xDMA seeds 0xFBFF
    l16 = np.zeros(Br, dtype=f16)
    o = np.zeros((s2_m, N, meshRow, meshCol), dtype=np.int64)
    t = np.zeros((s2_m, N, meshRow, meshCol), dtype=np.float64)
    moved, corr_min = 0, 1.0

    for j in range(NKV):
        # The D port's RNE(S * 2^-shift): exact power-of-two scaling in float64, one rounding.
        S16 = (_scores(Ks[j], B, M, N, K, meshRow, tileSize, meshCol).astype(np.float64)
               * 2.0 ** -shift).astype(f16)
        assert np.isfinite(S16).all(), "a score overflows FP16 even after the converter's shift"
        mnew = np.maximum(m16, S16.max(axis=0))
        if j > 0:
            moved += int((mnew > m16).sum())
        with np.errstate(over="ignore"):
            d16 = (m16.astype(f32) - mnew.astype(f32)).astype(f16)
            corr16 = np.exp(a32 * d16.astype(f32)).astype(f16)
        x16 = (S16.astype(f32) - mnew.astype(f32)).astype(f16)
        p16 = np.exp(a32 * x16.astype(f32)).astype(f16)
        rsum16 = p16.astype(f32).sum(axis=0).astype(f16)
        lsc16 = (corr16.astype(f32) * l16.astype(f32)).astype(f16)
        l16 = (rsum16.astype(f32) + lsc16.astype(f32)).astype(f16)
        pq = q127 * p16.astype(f32)
        p8 = np.clip(np.rint(np.clip(pq, -128.0, 128.0)), -127, 127).astype(np.int64)

        # P^T as the canonical B operand block_gemm_golden_model reads: [n][k][col][size],
        # element (n, k, c, s) = P8(key = 4k + s, query = 16n + c).
        b_can = p8.reshape(s2_k, tileSize, N, meshCol).transpose(2, 0, 3, 1).reshape(-1)
        zc = np.zeros(s2_m * N * meshRow * meshCol, dtype=np.int64)
        pv = np.asarray(block_gemm_golden_model(
            s2_m, s2_k, N, meshRow, tileSize, meshCol, Vs[j], b_can, 0, 0, zc),
            dtype=np.int64).reshape(o.shape)
        pv_abs = np.asarray(block_gemm_golden_model(
            s2_m, s2_k, N, meshRow, tileSize, meshCol, np.abs(Vs[j]), b_can, 0, 0, zc),
            dtype=np.int64).reshape(o.shape)

        # O^T's column is the query, so corr is a per-column factor. PV(0) masks its C (the
        # accumulator seed is zero); PV(j>=1) scales C by corr_j on the read path. x * f is
        # exact in float64 (31 x 11 bits), so np.rint IS the scaler's round-to-nearest-even.
        fcol = corr16.astype(np.float64).reshape(N, meshCol)[None, :, None, :]
        if j == 0:
            o = pv
            t = pv_abs.astype(np.float64)
        else:
            o = np.clip(np.rint(o * fcol), -2**31, 2**31 - 1).astype(np.int64) + pv
            t = t * fcol + pv_abs
            corr_min = min(corr_min, float(corr16.astype(np.float64).min()))
        m16 = mnew

        # ---- the layout claim, checked here rather than trusted ----------------------
        # Build the bytes the interleave quantiser writes, then walk them the way PV's B
        # reader does with its k-major strides {N*64, 64}. The result must be b_can.
        if j == 0:
            mem = np.zeros(Bc * Br, dtype=np.int64)
            for key in range(Bc):
                for q in range(Br):
                    mem[p8_interleave_offset(key, q, N, tileSize, meshCol)] = p8[key, q]
            blk = tileSize * meshCol
            walked = np.zeros((N, s2_k, meshCol, tileSize), dtype=np.int64)
            for kk in range(s2_k):
                for nn in range(N):
                    base = kk * (N * blk) + nn * blk
                    walked[nn, kk] = mem[base:base + blk].reshape(meshCol, tileSize)
            assert (walked.reshape(-1) == b_can).all(), \
                "interleaved P8 walked k-major is not the B operand P^T"
            # ...while the plain 2:1 pack, read the same way, is not: that is why the interleave
            # exists.
            assert not (p8.reshape(-1) == b_can).all()

    # ---- the last tile's softmax, sampled -------------------------------------------
    stride = max(1, Bc // 16)
    beats = list(range(0, Bc, stride))
    frac = pq - np.floor(pq)
    p8_dc = np.where(np.abs(frac - 0.5) < 0.1, -1, p8).astype(np.int16)
    p8s = p8_dc[beats].reshape(-1)
    # Per-query count of P8 == 127 over the whole last tile: the row maximum always
    # exponentiates to exactly 1.0, so the count is >= 1 and pins the subtract, the exp,
    # the 127 scale and -- through P8_OFF -- the interleaved layout. Rows with a lane near
    # the 126.5 threshold are don't-care (-1).
    c127 = (p8 == 127).sum(axis=0).astype(np.int16)
    c127 = np.where((np.abs(pq - 126.5) < 0.1).any(axis=0), -1, c127).astype(np.int16)

    # O tolerance: 1/64 of sum|terms| plus two P8 steps of |V| <= 16. One LUT-rounding flip
    # of a P8 moves an element by |V| <= 16; a MISSING rescale moves it by (1 - corr) * O.
    o_tol = (np.ceil(t / 64.0) + 32).astype(np.int64)

    # ---- then put it where the port actually writes it -------------------------
    #
    # The C/D spatial map is chosen so the FP16 score tile lands row-major -- one key per
    # 64 B beat, which is what the LANEWISE reduce needs. C and D share those strides and
    # the INT32 side is twice as wide, so O comes out PERMUTED. That is deliberate (C and
    # D permute identically, so O += P.V still accumulates against itself) but it means a
    # golden in canonical block order is not comparable.
    #
    # The map is derived from the emitted descriptors, not assumed: the array serialises a
    # meshRow x meshCol block into chunks of serial_c_d_width, each chunk into 16 channels
    # of bankWidth, and the AGU places channel i at sl0*(i%4) + sl1*((i/4)%4). The same
    # formula run at FP16 has to come out exactly row-major -- that is the assert below,
    # and it is what says the model of the serialisation is right rather than plausible.
    serial_c_d = _acc(kwargs)["snax_versacore_serial_c_d_width"]
    slstride = [bankWidth // 8, N * meshCol * 16 // 8]
    ts0, ts1 = serial_c_d * N // 8, meshCol * 16 // 8

    # How the AGU places channel i. The port declares its own spatial grouping, so read it
    # rather than assume one: [4, 4] interleaves N blocks (needed when Nu < Br), [16] is a
    # plain contiguous transaction (available once Nu == Br). Getting this wrong is what the
    # FP16 assert below catches.
    sbounds = [
        int(b)
        for b in kwargs["snax_versacore_streamer_template"]["data_reader_writer_params"][
            "spatial_bounds"
        ][0]
    ]

    def chan_off(ch):
        off, rem = 0, ch
        for d, b in enumerate(sbounds):
            off += slstride[d] * (rem % b)
            rem //= b
        return off

    def scatter(width, blocks, ts2, elem_bytes):
        """byte offset of every (block, row, col) element, in the port's order."""
        per_chunk = serial_c_d // width          # elements in one serialised chunk
        per_chan  = bankWidth // width           # elements in one channel
        out = {}
        for mm in range(blocks):
            for nn in range(N):
                for r in range(meshRow):
                    for c in range(meshCol):
                        e = r * meshCol + c
                        ch = (e % per_chunk) // per_chan
                        out[(mm, nn, r, c)] = (
                            mm * ts2 + nn * ts1 + (e // per_chunk) * ts0
                            + chan_off(ch)
                            + (e % per_chan) * elem_bytes
                        )
        return out

    # self-test: at FP16 the same model must give plain [key][query] row-major
    fp16_map = scatter(16, 1, N * 16 * meshRow * meshCol // 8, 2)
    for (mm, nn, r, c), b in fp16_map.items():
        want = ((mm * meshRow + r) * (N * meshCol) + nn * meshCol + c) * 2
        assert b == want, "D32 address model disagrees with the FP16 row-major layout"

    ts2 = N * output_data_width * meshRow * meshCol // 8
    o_mem = np.zeros(o.size, dtype=np.int64)
    tol_mem = np.zeros(o.size, dtype=np.int64)
    seen = np.zeros(o.size, dtype=bool)
    for (mm, nn, r, c), b in scatter(output_data_width, s2_m, ts2, 4).items():
        w = b // 4
        assert b % 4 == 0 and not seen[w], "D32 INT32 address map is not a bijection"
        seen[w] = True
        o_mem[w] = o[mm, nn, r, c]
        tol_mem[w] = o_tol[mm, nn, r, c]
    assert seen.all(), "D32 INT32 address map does not cover the output"

    def bits(x):
        return np.ascontiguousarray(x, dtype=np.float16).view(np.uint16).reshape(-1)

    return "\n".join([
        "// ---- ATTENTION GOLDEN ------------------------------------------------------",
        "// FP16 bit patterns, compared by ULP distance: the check core has no FPU.",
        "// a = %r = 2^-%d (converter) x %r (exp). S16 = RNE(S_int * 2^-%d) on the D port,"
        % (float(kwargs["SCORE_SCALE"]), shift, float(a32), shift),
        "// then P = exp(a' * (S16 - m)) in StreamMap.",
        "#define D32_FP16_SHIFT %d  // Int32ToFp16 csr(1): power-of-two output scale" % shift,
        "#define SCORE_SCALE_BITS 0x%08Xu  // a' = %r, FP32: the part of a the exp applies"
        % (int(np.array(a32).view(np.uint32)), float(a32)),
        "#define PGOLD_STRIDE %d  // every Nth beat of P is checked" % stride,
        "#define PGOLD_NBEATS %d" % len(beats),
        "// (query, tile >= 1) pairs whose running max moved, i.e. where corr != 1 and the",
        "// O rescale is load-bearing; and the smallest corr applied.",
        "#define GOLD_MAX_MOVES %d  // of %d" % (moved, Br * (NKV - 1)),
        "#define GOLD_CORR_MIN_PERMILLE %d" % int(corr_min * 1000),
        "",
        "// m after the last tile (the global max) and l after the last tile",
        format_vector_definition("uint16_t", "m_golden", bits(m16)),
        "",
        format_vector_definition("uint16_t", "l_golden", bits(l16)),
        "",
        "// the LAST tile's row sum",
        format_vector_definition("uint16_t", "rowsum_golden", bits(rsum16)),
        "",
        "// the last tile's P8 = sat(rne(127 P)) on every PGOLD_STRIDE-th key, [key][query],",
        "// -1 = within 0.1 of a rounding boundary (don't care)",
        format_vector_definition("int16_t", "p8_golden", p8s),
        "",
        "// P8 == 127 count per query row over the whole last tile, -1 = don't care",
        format_vector_definition("int16_t", "p8max_golden", c127),
        "",
        "// O after NKV tiles WITH the online rescale, in the order the D32 port writes it,",
        "// and the per-element tolerance |O - O_golden| <= o32_tol.",
        format_vector_definition("int32_t", "o32_golden", o_mem.astype(np.int32)),
        "",
        format_vector_definition("int32_t", "o32_tol", tol_mem.astype(np.int32)),
    ])


def emit_geometry_section(**kwargs):
    """The tile geometry as compile-time constants, derived from the shape and the mesh.

    The kernel takes BR/BC/DHEAD/NKV from here rather than defining its own, so it
    cannot disagree with the descriptors it is handed.
    """
    meshRow, tileSize, meshCol = _mesh(kwargs)
    d = kwargs["K"] * tileSize
    rows = [
        # The array, read from the cluster cfg. It reaches the kernel through HERE, the
        # same place the descriptors come from, which is what makes the kernel's
        # "descriptors vs geometry" check meaningful.
        ("meshRow", meshRow, "VersaCore Mu: output rows    per array pass"),
        ("tileSize", tileSize, "VersaCore Ku: contraction    per array pass"),
        ("meshCol", meshCol, "VersaCore Nu: output columns per array pass"),
        ("BR", kwargs["N"] * meshCol, "query rows     = N*meshCol"),
        ("BC", kwargs["M"] * meshRow, "key columns    = M*meshRow (the tiling knob)"),
        ("DHEAD", d, "head dimension = K*tileSize (a MODEL property, not a knob)"),
        ("NKV", kwargs["NKV"], "KV tiles streamed through the software pipeline"),
    ]
    w = max(len(str(v)) for _, v, _ in rows)
    n = max(len(name) for name, _, _ in rows)
    body = "\n".join(
        "#define %-*s %*d  // %s" % (n, name, w, value, note)
        for name, value, note in rows
    )
    return (
        "// ---- TILE GEOMETRY -------------------------------------------------------"
        "---\n" + body + "\n"
    )


# Add stdint.h header
def emit_header_file(**kwargs):
    emit_str = "#include <stdint.h>\n\n"
    emit_str += emit_geometry_section(**kwargs) + "\n"
    emit_str += emit_versacore_data(**kwargs)
    return emit_str


MIN = -128
MAX = 127

bankWidth = 64
input_data_width = 8
output_data_width = 32
quantized_output_data_width = 8


def emit_matmul_data(**kwargs):

    meshRow, tileSize, meshCol = _mesh(kwargs)
    # C and D are Mu*Nu INT32 = 8192 b in the array and reach TCDM over a 2048 b port, so
    # every output block is four serialised chunks. One number drives both descriptors.
    serial_c_d_width = _acc(kwargs)["snax_versacore_serial_c_d_width"]

    # matmul settings
    data_str = []

    data_str += [format_scalar_definition("int32_t", "Batch", 1)]
    data_str += [format_scalar_definition("int32_t", "M", kwargs["M"])]
    data_str += [format_scalar_definition("int32_t", "K", kwargs["K"])]
    data_str += [format_scalar_definition("int32_t", "N", kwargs["N"])]

    # One spatial unrolling and one data type in this cfg -- see _mesh().
    data_str += [format_scalar_definition("int32_t", "array_shape", 0)]
    data_str += [format_scalar_definition("uint32_t", "data_type", 0)]

    data_str += [format_scalar_definition("int32_t", "Aslstride0", bankWidth / 8)]
    data_str += [format_scalar_definition("int32_t", "Atlbound0", kwargs["K"])]
    data_str += [
        format_scalar_definition(
            "int32_t", "Atlstride0", input_data_width * tileSize * meshRow / 8
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Atlbound1", kwargs["N"])]
    data_str += [format_scalar_definition("int32_t", "Atlstride1", 0)]
    data_str += [format_scalar_definition("int32_t", "Atlbound2", kwargs["M"])]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "Atlstride2",
            kwargs["K"] * input_data_width * tileSize * meshRow / 8,
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Atlbound3", 1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride3", 0)]
    data_str += [format_scalar_definition("int32_t", "Atlbound4", 1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride4", 0)]
    data_str += [format_scalar_definition("int32_t", "Atlbound5", 1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride5", 0)]

    data_str += [format_scalar_definition("int32_t", "Bslstride0", bankWidth / 8)]
    data_str += [format_scalar_definition("int32_t", "Btlbound0", kwargs["K"])]
    data_str += [
        format_scalar_definition(
            "int32_t", "Btlstride0", input_data_width * tileSize * meshCol / 8
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Btlbound1", kwargs["N"])]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "Btlstride1",
            kwargs["K"] * input_data_width * tileSize * meshCol / 8,
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Btlbound2", kwargs["M"])]
    data_str += [format_scalar_definition("int32_t", "Btlstride2", 0)]

    # -----------------------------------------------------------
    # streamer c32 settings
    # -----------------------------------------------------------
    # The C/D port's 16 channels are grouped [4, 4] (see the cluster cfg): channel i sits
    # at sl0*(i % 4) + sl1*((i / 4) % 4). Four channels carry 32 B -- one key's meshCol
    # FP16 scores -- and the eight groups step by a WHOLE key row of Br = N*meshCol FP16,
    # so the two N blocks INTERLEAVE and a 64 B beat is 32 queries of ONE key. That is the
    # layout snax-flashattn.c reduces over; under a contiguous [8, 4] map a beat would be
    # 16 queries x 2 keys and the LANEWISE rowmax would be meaningless.
    #
    #   spatial 0 : 4 channels x bankWidth/8   = 32 B   one key, meshCol FP16 scores
    #   spatial 1 : 8 groups   x Br*2          = 64 B   pitch: one whole key row
    #   temporal 0: serial chunk               = 512 B  the rows one transaction covers
    #   temporal 1: the OTHER N block          = 32 B   the interleave offset
    #   temporal 2: the next M block           = 2048 B
    #
    # BOTH strides must be emitted: the port declares two, so a 1-element array feeds the
    # second from off the end of the caller's stack and 8 of the 16 channels address
    # garbage.
    data_str += [format_scalar_definition("int32_t", "Cslstride0", bankWidth / 8)]
    data_str += [
        format_scalar_definition(
            "int32_t", "Cslstride1", kwargs["N"] * meshCol * 16 / 8
        )
    ]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "Ctlbound0",
            output_data_width * meshRow * meshCol / serial_c_d_width,
        )
    ]
    data_str += [
        format_scalar_definition(
            "int32_t", "Ctlstride0", serial_c_d_width * kwargs["N"] / 8
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Ctlbound1", kwargs["N"])]
    data_str += [
        format_scalar_definition("int32_t", "Ctlstride1", meshCol * 16 / 8)
    ]
    data_str += [format_scalar_definition("int32_t", "Ctlbound2", kwargs["M"])]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "Ctlstride2",
            kwargs["N"] * output_data_width * meshRow * meshCol / 8,
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Ctlbound3", 1)]
    data_str += [format_scalar_definition("int32_t", "Ctlstride3", 0)]

    # -----------------------------------------------------------
    # streamer d32 settings
    # -----------------------------------------------------------
    # The C/D port's 16 channels are grouped [4, 4] (see the cluster cfg): channel i sits
    # at sl0*(i % 4) + sl1*((i / 4) % 4). Four channels carry 32 B -- one key's meshCol
    # FP16 scores -- and the eight groups step by a WHOLE key row of Br = N*meshCol FP16,
    # so the two N blocks INTERLEAVE and a 64 B beat is 32 queries of ONE key. That is the
    # layout snax-flashattn.c reduces over; under a contiguous [8, 4] map a beat would be
    # 16 queries x 2 keys and the LANEWISE rowmax would be meaningless.
    #
    #   spatial 0 : 4 channels x bankWidth/8   = 32 B   one key, meshCol FP16 scores
    #   spatial 1 : 8 groups   x Br*2          = 64 B   pitch: one whole key row
    #   temporal 0: serial chunk               = 512 B  the rows one transaction covers
    #   temporal 1: the OTHER N block          = 32 B   the interleave offset
    #   temporal 2: the next M block           = 2048 B
    #
    # BOTH strides must be emitted: the port declares two, so a 1-element array feeds the
    # second from off the end of the caller's stack and 8 of the 16 channels address
    # garbage.
    data_str += [format_scalar_definition("int32_t", "D32slstride0", bankWidth / 8)]
    data_str += [
        format_scalar_definition(
            "int32_t", "D32slstride1", kwargs["N"] * meshCol * 16 / 8
        )
    ]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "D32tlbound0",
            output_data_width * meshRow * meshCol / serial_c_d_width,
        )
    ]
    data_str += [
        format_scalar_definition(
            "int32_t", "D32tlstride0", serial_c_d_width * kwargs["N"] / 8
        )
    ]
    data_str += [format_scalar_definition("int32_t", "D32tlbound1", kwargs["N"])]
    data_str += [
        format_scalar_definition("int32_t", "D32tlstride1", meshCol * 16 / 8)
    ]
    data_str += [format_scalar_definition("int32_t", "D32tlbound2", kwargs["M"])]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "D32tlstride2",
            kwargs["N"] * output_data_width * meshRow * meshCol / 8,
        )
    ]
    data_str += [format_scalar_definition("int32_t", "D32tlbound3", 1)]
    data_str += [format_scalar_definition("int32_t", "D32tlstride3", 0)]

    # No D8 descriptors: VersaCore has ONE output port, and the quantised path is a write
    # extension on it rather than a writer of its own.

    # -----------------------------------------------------------
    delta_local_a = 0
    delta_local_b = (
        kwargs["K"] * kwargs["M"] * (meshRow * tileSize * input_data_width / 8)
    )
    delta_local_b = align_wide_addr(delta_local_b)
    delta_local_c = delta_local_b + kwargs["K"] * kwargs["N"] * (
        meshCol * tileSize * input_data_width / 8
    )
    delta_local_c = align_wide_addr(delta_local_c)
    delta_local_d32 = delta_local_c + kwargs["M"] * kwargs["N"] * (
        meshRow * meshCol * output_data_width / 8
    )
    delta_local_d32 = align_wide_addr(delta_local_d32)
    data_str += [format_scalar_definition("int32_t", "delta_local_a", delta_local_a)]
    data_str += [format_scalar_definition("int32_t", "delta_local_b", delta_local_b)]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "delta_local_c",
            delta_local_c,
        )
    ]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "delta_local_d32",
            delta_local_d32,
        )
    ]
    # -----------------------------------------------------------
    # Test Data generation
    # -----------------------------------------------------------

    # No zero-point subtraction: attention has none, and the kernel programs
    # gen_subtraction_config(0, 0) on every dispatch.

    # NKV DISTINCT key tiles and NKV distinct value tiles. With one tile replayed NKV
    # times the running max stops moving after tile 0, corr is exp(0) = 1 for ever and
    # the online rescale is never exercised. A holds the K tiles back to back, V the V
    # tiles; the kernel streams
    # tile j from offset j * Bc * d of each. V's bytes are read by PV as the shape-2 A
    # operand V^T, and the golden reinterprets them the same way.
    nkv = int(kwargs.get("NKV", 1))
    kv_shape = (kwargs["M"], kwargs["K"], meshRow, tileSize)
    A = np.random.randint(MIN, MAX, size=(nkv,) + kv_shape).reshape(-1)
    V = np.random.randint(MIN, MAX, size=(nkv,) + kv_shape).reshape(-1)
    B = np.random.randint(
        MIN, MAX, size=(kwargs["K"], kwargs["N"], tileSize, meshCol)
    ).reshape(-1)

    # FULL-RANGE INT8. The D-port converter applies 2^-D32_FP16_SHIFT while it rounds (see
    # score_scale_split), so even the worst-case score, 127^2 * d, converts to a finite FP16.
    A = A.astype(np.int8)
    V = V.astype(np.int8)
    B = B.astype(np.int8)

    # 64-BYTE ALIGNED, because these are xDMA sources. The iDMA copies bytes at any
    # alignment, but the xDMA reader issues eight 8-byte channels per beat and needs
    # its base aligned to the beat. Unaligned, it still moves the right number of
    # bytes at the right rate and only the DATA is wrong: m, P8 and rowsum stay
    # bit-exact (they come from K, loaded by the iDMA) while O, which comes from V,
    # does not.
    #
    # V FIRST, because the xDMA reads it and the xDMA's view of main memory is only
    # 512 KiB wide: the TB endpoint's TCDMAddrWidth is 19, so an address past
    # 0x8008_0000 silently WRAPS to the bottom of DRAM, onto .text. Emitted after A, V's
    # last tile would cross that edge, and PV would multiply code bytes into O.
    # The iDMA (K tiles 1+) has full reach; the xDMA also carries half of K tile 0, which
    # is why A follows directly. The kernel asserts both ranges at run time.
    data_str += [format_vector_definition("int8_t", "V", V, alignment=64)]
    data_str += [format_vector_definition("int8_t", "A", A, alignment=64)]
    data_str += [format_vector_definition("int8_t", "B", B, alignment=64)]

    enabled_channel_CSR_num = int(math.ceil(
        (meshRow * meshCol) * output_data_width / bankWidth / 32
    ))

    broadcast_C = kwargs["broadcast_C"] == 1 and kwargs["channel_en_C"] == 1
    disable_C = kwargs["broadcast_C"] == 0 and kwargs["channel_en_C"] == 0
    enable_full_C = kwargs["broadcast_C"] == 0 and kwargs["channel_en_C"] == 1

    assert broadcast_C or disable_C or enable_full_C, "Invalid C settings"

    if broadcast_C == 1:
        C = np.random.randint(MIN, MAX, size=(kwargs["M"], kwargs["N"], 1, meshCol))
        C = np.repeat(C, repeats=meshRow, axis=1).reshape(-1)
    elif enable_full_C == 1:
        C = np.random.randint(
            MIN, MAX, size=(kwargs["M"], kwargs["N"], meshRow, meshCol)
        ).reshape(-1)
    else:
        C = np.random.randint(
            0, 1, size=(kwargs["M"], kwargs["N"], meshRow, meshCol)
        ).reshape(-1)

    if broadcast_C == 1:
        assert meshCol * output_data_width % bankWidth == 0
        # Note: if C is hanged to wide ports, the mimimum number of bits to enable
        # is multipliers of 8 (8 narrow channels equal to 1 wide channel)
        channel_en_C_1_bits = int(
            (meshCol * output_data_width / bankWidth + 7) // 8 * 8
        )
        # Generate the elements
        channel_en_C = [0] * enabled_channel_CSR_num  # Initialize with zeros

        for i in range(channel_en_C_1_bits):
            element_index = i // 32  # Determine which element to modify
            bit_position = i % 32  # Position within the element
            if element_index < enabled_channel_CSR_num:
                channel_en_C[element_index] |= 1 << (bit_position)

        # Convert elements to integers
        channel_en_C = [int(x) for x in channel_en_C][::-1]  # Reverse the list
    elif enable_full_C == 1:
        channel_en_C = [((1 << 32) - 1) for i in range(enabled_channel_CSR_num)]
    else:
        channel_en_C = [0 for i in range(enabled_channel_CSR_num)]
    data_str += [
        "int32_t channel_en_C[] = { " + ", ".join(map(str, channel_en_C)) + " };"
    ]

    data_str += [
        format_scalar_definition("int32_t", "broadcast_C", kwargs["broadcast_C"])
    ]
    data_str += [format_vector_definition("int32_t", "C", C)]

    # No operand transposer on this cluster, so no permutation of A or B here either.
    # FlashAttention transposes ALGEBRAICALLY -- S^T = K.Q^T is the same GEMM with its
    # operands swapped -- and never enabled the hardware one. See the streamer template.

    return data_str, A, V, B


def emit_versacore_data(**kwargs):
    data_str, A, V, B = emit_matmul_data(**kwargs)

    # No rescale epilogue and no raw-matmul golden. VersaCore has no rescale unit and this
    # cluster gives its write path no rescale extension, so there are no zero-point,
    # shift/multiplier or rounding CSRs to feed. The kernel checks the SOFTMAX, against
    # emit_attention_golden(), rather than the matmul, so a D32 vector would be dead
    # weight in data.h.

    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_A", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_B", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_C", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_D32", 0)]

    # Shape 2 regenerates the header only to harvest its descriptors; it has no golden.
    if not kwargs.get("_descriptors_only", False):
        tile = A.size // int(kwargs["NKV"])
        Ks = [A[j * tile:(j + 1) * tile] for j in range(int(kwargs["NKV"]))]
        Vs = [V[j * tile:(j + 1) * tile] for j in range(int(kwargs["NKV"]))]
        data_str += [emit_attention_golden(Ks, Vs, B, **kwargs)]

    data_str = "\n\n".join(data_str)

    return data_str


def emit_shape2_section(param, merged_config):
    """The SECOND matmul shape's streamer descriptors, prefixed S2_, appended to data.h.

    FlashAttention's two matmuls stop being the same shape once d != Bc:

        matmul 1  S^T = K.Q^T     M1*meshRow = Bc,  K1*tileSize = d   <- params.hjson
        matmul 2  O^T = V^T.P^T   M2*meshRow = d,   K2*tileSize = Bc  <- here

    M and K swap roles, so every temporal bound and stride changes with them. The second
    shape is DERIVED from the first rather than configured separately -- otherwise the two
    drift apart the moment the tile is retuned:

        M2 = K1 * tileSize / meshRow      (since K1*tileSize = d)
        K2 = M1 * meshRow  / tileSize     (since M1*meshRow  = Bc)

    The descriptors themselves come from the SAME generator as shape 1, with M and K
    swapped, because hand-deriving bounds and strides is where a silent stride error would
    live. Only the scalar descriptors are kept: shape 2 reuses shape 1's buffers, it just
    walks them differently.
    """
    mesh_row, tile_size, _ = _mesh(merged_config)

    m1, n1, k1 = int(param["M"]), int(param["N"]), int(param["K"])
    d, bc = k1 * tile_size, m1 * mesh_row
    assert d % mesh_row == 0, f"d={d} is not a multiple of meshRow={mesh_row}"
    assert bc % tile_size == 0, f"Bc={bc} is not a multiple of tileSize={tile_size}"
    m2, n2, k2 = d // mesh_row, n1, bc // tile_size

    shape2 = {**merged_config, "M": m2, "N": n2, "K": k2, "_descriptors_only": True}
    keep = re.compile(
        r"^int32_t (M|N|K|[ABCD][0-9]*[a-z]*(?:sl|tl)(?:bound|stride)[0-9]+) = "
    )
    lines = [
        "",
        "// ---- SECOND MATMUL SHAPE " + "-" * 52,
        f"// O^T = V^T.P^T : M={m2} N={n2} K={k2}   (d={d}, Bc={bc}, derived from shape 1 above).",
        "// Switched in per dispatch by gemm_set_shape(); only the bounds and strides that",
        "// depend on M and K differ from shape 1.",
    ]
    lines += [
        ln.replace("int32_t ", "static const int32_t S2_", 1)
        for ln in emit_header_file(**shape2).splitlines()
        if keep.match(ln)
    ]
    return "\n".join(lines)


def main():
    # Parsing cmd args
    parser = argparse.ArgumentParser(description="Generate data for kernels")
    parser.add_argument(
        "--swcfg",
        type=pathlib.Path,
        required=True,
        help="Select param config file kernel",
    )
    parser.add_argument(
        "--hwcfg",
        type=pathlib.Path,
        required=True,
        help="Select hardware config file kernel",
    )
    args = parser.parse_args()

    # Load param config file
    with args.swcfg.open() as f:
        param = hjson.loads(f.read())

    # Load hardware config file
    with args.hwcfg.open() as f:
        hw = hjson.loads(f.read())

    # Merge dictionaries (hw overrides param in case of conflicts)
    merged_config = {**param, **hw}

    # Emit header file: shape 1's full data set, then shape 2's descriptors appended
    # to the SAME file. One generator, one output, one #include.
    print(emit_header_file(**merged_config))
    print(emit_shape2_section(param, merged_config))


if __name__ == "__main__":

    main()
