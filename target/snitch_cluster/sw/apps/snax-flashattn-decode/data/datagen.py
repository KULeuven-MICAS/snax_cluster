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
    simd_beat_bytes,
    simd_fp16_lanes,
)  # noqa E402

np.random.seed(42)


def _acc(kwargs):
    return kwargs["snax_versacore_core_template"]["snax_acc_cfg"][0]


def _mesh(kwargs):
    """meshRow, tileSize, meshCol -- VersaCore's (Mu, Ku, Nu) spatial unrolling.

    VersaCore carries one triple per [data type][array shape] and selects between them at
    run time with two CSRs. The kernel runs data type 0, array shape 0 -- it writes both
    CSRs as 0 -- so the mesh is that entry. The kernel never selects the cluster's other
    shapes; the assert keeps shape 0 a matmul shape, since a one-row shape there would emit
    descriptors for the wrong mesh.
    """
    unrolling = _acc(kwargs)["snax_versacore_spatial_unrolling"]
    mesh = tuple(unrolling[0][0])
    assert mesh[0] > 1, (
        "this kernel emits array_shape = data_type = 0, and the cfg's shape 0 is %s" % (mesh,)
    )
    return mesh


def _simd_cfg(kwargs):
    for core in kwargs["cluster"]["hives"][0]["cores"]:
        if isinstance(core, dict) and "snax_simd_cfg" in core:
            return core["snax_simd_cfg"]
    raise ValueError("this cluster has no SIMD core")


def _reduce_partials(kwargs):
    """Rotating FP32 partials per lane in the SIMD core's StreamReduce: its accumulate latency.

    Beat k of a row accumulates into partial k mod P, P = 2*fpPipe + 1, and the row's result
    folds the partials as a balanced tree -- that order is what the rowsum golden follows.
    """
    for ext in _simd_cfg(kwargs)["reader_extensions"]:
        if "HasSimdStreamReduce" in ext:
            return 2 * int(ext["HasSimdStreamReduce"].get("fpPipe", 1)) + 1
    raise ValueError("the SIMD core has no StreamReduce")


def fa_layout(kwargs):
    """The kernel's score-tile and P8 layouts, which the descriptors and the goldens share.

    ONE SIMD BEAT IS ONE KEY OF ONE ARRAY COLUMN BLOCK: meshCol queries, one FP16 lane each.
    The kernel requires SIMD_FP16_LANES == meshCol, so a beat is exactly one row of an array
    output block, and Br = N*meshCol queries are N beats per key. The score tile S^T is
    therefore stored as N PLANES, plane n = queries n*meshCol.., one beat per key, with a
    spare beat on either side:

        plane n:  [ latch ][ key 0 ][ key 1 ] .. [ key Bc-1 ][ m_old ]      s_pitch bytes

    The latch is the sticky-B operand the epilogue needs immediately before its data (-m_new
    of half n); m_old is read as the row's last beat by the rowmax, so that one lanewise MAX
    gives max(m_old, rowmax) directly. The D32 port writes the key rows (see cd_map).

    P8 is laid out n-major, one half per epilogue task, each followed by its two tail beats:

        half n:   [ B block (k=0, n) ] .. [ B block (k=Bc/4-1, n) ][ rowsum ][ corr*l_old ]

    so PV's B reader walks k in tileSize*meshCol = 64 B steps, the rate its A reader walks V.
    """
    meshRow, tileSize, meshCol = _mesh(kwargs)
    lanes, beat = simd_fp16_lanes(kwargs), simd_beat_bytes(kwargs)
    assert lanes == meshCol, (
        "one SIMD beat must be one array output row: %d FP16 lanes against meshCol = %d"
        % (lanes, meshCol))
    Bc = kwargs["M"] * meshRow
    return {
        "beat": beat,
        "lanes": lanes,
        "s_pitch": (Bc + 2) * beat,
        "p_pitch": Bc * meshCol + 2 * beat,
    }


def cd_map(kwargs):
    """The C/D port's address map: spatial strides and the INT32 and FP16 temporal walks.

    The port's 16 channels are grouped by the cfg's spatial_bounds (channel i at
    sl0*(i % b0) + sl1*((i / b0) % b1)), with sl1 = b0 banks, so one 1024 b transaction is
    128 contiguous bytes: 4 key rows of meshCol FP16 scores, or 2 rows of meshCol INT32.
    Temporal loop 0 steps those chunks (contiguous in both widths), loop 1 the N column
    blocks, loop 2 the M row blocks:

        INT32 (C and D, PV's O^T)  block (m, n) at m*N*1 KiB + n*1 KiB, row-major inside
        FP16  (D, QK's S^T)        block (m, n) at n*s_pitch + m*meshRow*beat: plane n

    C and D share the INT32 map, so O += P.V accumulates against itself. The converter halves
    the beats D emits, so the FP16 walk has its own bound 0, n stride and m stride; the kernel
    writes them per dispatch (gemm_d32_emit_fp16).
    """
    meshRow, tileSize, meshCol = _mesh(kwargs)
    serial = _acc(kwargs)["snax_versacore_serial_c_d_width"]
    sbounds = [
        int(b)
        for b in kwargs["snax_versacore_streamer_template"]["data_reader_writer_params"][
            "spatial_bounds"
        ][0]
    ]
    lay = fa_layout(kwargs)
    blk32 = meshRow * meshCol * output_data_width // 8
    m = {
        "sbounds": sbounds,
        "sl": [bankWidth // 8, sbounds[0] * bankWidth // 8],
        "i32": {
            "b0": output_data_width * meshRow * meshCol // serial,
            "ts": [serial // 8, blk32, kwargs["N"] * blk32],
        },
        "f16": {
            "b0": 16 * meshRow * meshCol // serial,
            "ts": [serial // 8, lay["s_pitch"], meshRow * lay["beat"]],
        },
    }
    assert m["i32"]["ts"][0] == m["f16"]["ts"][0], "the kernel does not rewrite D's chunk stride"
    return m


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


def p8_interleave_offset(key, q, tileSize, meshCol, p_pitch):
    """Byte offset of P8(key, query) in a P8 buffer, as the kernel's epilogue writes it.

    One epilogue task quantises one N half (queries n*meshCol..) and writes its B blocks
    (k, n) one after another, k = key/tileSize: n-major, half n at n*p_pitch, block k at
    k*tileSize*meshCol, and inside the block one tileSize-byte atom per query. The kernel's
    P8_OFF() is this formula; emit_attention_golden() checks it against the quantiser's
    INTERLEAVE4 output order and against the walk PV's B reader performs.
    """
    n, c = q // meshCol, q % meshCol
    return n * p_pitch + (key // tileSize) * tileSize * meshCol + c * tileSize + key % tileSize


def lanewise_sum(x16, partials):
    """StreamReduce ADD|LANEWISE over the beats of x16 [beats, lanes], in the core's order.

    Beat k accumulates into FP32 partial k mod `partials` (one rounding per add: the FMA's
    product x*1 is exact), and the row's result folds the partials it wrote as a balanced
    pairwise tree, ((p0 + p1) + p2) for three, before one narrowing to FP16.
    """
    f32 = np.float32
    x = x16.astype(f32)
    acc = [None] * partials
    for k in range(x.shape[0]):
        p = k % partials
        acc[p] = x[k].copy() if acc[p] is None else (acc[p] + x[k]).astype(f32)
    vals = [a for a in acc if a is not None]
    while len(vals) > 1:
        vals = [(vals[i] + vals[i + 1]).astype(f32) if i + 1 < len(vals) else vals[i]
                for i in range(0, len(vals), 2)]
    return vals[0].astype(np.float16)


def emit_attention_golden(Ks, Vs, B, **kwargs):
    """A float model of the WHOLE online softmax over NKV distinct KV tiles.

    Every KV tile has its own K and V, so the running maximum moves and corr is not 1.
    The model follows the hardware's sequence at the precision each stage works in:

        S16    exact INT32 out of the mesh, RNE to FP16 on the D32 port
        m      max over keys of S16, running across tiles           (exact)
        corr   exp(a * fp16(m_old - m_new))                        EW0 ADD -> Map EXP
        P      exp(a * fp16(S16 - m_new)), stored FP16 in-chain     EW0 ADD -> Map EXP
        rowsum sum over keys of P in FP32 partials, narrowed once   Reduce TAP (lanewise_sum)
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
    lay = fa_layout(kwargs)
    lanes, beat, p_pitch = lay["lanes"], lay["beat"], lay["p_pitch"]
    partials = _reduce_partials(kwargs)

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
        rsum16 = lanewise_sum(p16, partials)
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
        # Build the bytes the epilogue writes -- one task per N half, its quantiser's
        # INTERLEAVE4 turning each group of 4 key beats (lanes = the half's queries) into two
        # output beats, out[q / (L/2)][4*(q % (L/2)) + b] = quant(beat_b[q]) at L lanes, the
        # writer laying the output beats down contiguously -- then walk them the way PV's B
        # reader does, strides {tileSize*meshCol, p_pitch}. The result must be b_can, and
        # every byte must sit where p8_interleave_offset() (the kernel's P8_OFF) says.
        if j == 0:
            mem = np.full(N * p_pitch, -999, dtype=np.int64)
            half = lanes // 2
            for nn in range(N):
                for g in range(Bc // tileSize):
                    for b in range(tileSize):
                        for q in range(lanes):
                            at = (nn * p_pitch + (2 * g + q // half) * beat
                                  + tileSize * (q % half) + b)
                            mem[at] = p8[tileSize * g + b, nn * meshCol + q]
            for key in range(0, Bc, 7):
                for q in range(Br):
                    assert mem[p8_interleave_offset(key, q, tileSize, meshCol, p_pitch)] \
                        == p8[key, q], "P8_OFF disagrees with the quantiser's output order"
            blk = tileSize * meshCol
            walked = np.zeros((N, s2_k, meshCol, tileSize), dtype=np.int64)
            for kk in range(s2_k):
                for nn in range(N):
                    base = nn * p_pitch + kk * blk
                    walked[nn, kk] = mem[base:base + blk].reshape(meshCol, tileSize)
            assert (walked.reshape(-1) == b_can).all(), \
                "interleaved P8 walked by PV's B reader is not the B operand P^T"
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
    # O^T is compared in the order the C/D port lays it down, derived from the emitted
    # descriptors (cd_map) rather than assumed: the array serialises a meshRow x meshCol
    # block into chunks of serial_c_d_width, each chunk into 16 channels of bankWidth, and
    # the AGU places channel i at sl0*(i % b0) + sl1*((i / b0) % b1). The same model run on
    # the FP16 walk has to come out as the kernel's score planes -- that is the self-test
    # below, and it is what says the model of the serialisation is right rather than
    # plausible.
    serial_c_d = _acc(kwargs)["snax_versacore_serial_c_d_width"]
    cdm = cd_map(kwargs)
    sbounds, slstride = cdm["sbounds"], cdm["sl"]

    def chan_off(ch):
        off, rem = 0, ch
        for d, b in enumerate(sbounds):
            off += slstride[d] * (rem % b)
            rem //= b
        return off

    def scatter(width, blocks, ts, elem_bytes):
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
                            mm * ts[2] + nn * ts[1] + (e // per_chunk) * ts[0]
                            + chan_off(ch)
                            + (e % per_chan) * elem_bytes
                        )
        return out

    # self-test: the FP16 walk lands key k of plane n at n*s_pitch + k*beat, query c of the
    # plane at 2c -- one SIMD beat per key, the layout the lanewise reductions need.
    for (mm, nn, r, c), b in scatter(16, M, cdm["f16"]["ts"], 2).items():
        want = nn * lay["s_pitch"] + (mm * meshRow + r) * beat + c * 2
        assert b == want, "D32 address model disagrees with the FP16 score planes"

    o_mem = np.zeros(o.size, dtype=np.int64)
    tol_mem = np.zeros(o.size, dtype=np.int64)
    seen = np.zeros(o.size, dtype=bool)
    for (mm, nn, r, c), b in scatter(output_data_width, s2_m, cdm["i32"]["ts"], 4).items():
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
        "#define PGOLD_STRIDE %d  // every Nth key of P is checked" % stride,
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
    # The score-plane and P8 layouts (fa_layout) and the D port's FP16 walk onto the planes
    # (cd_map): the kernel allocates its buffers and programs QK's D32 writer from these.
    lay, f16 = fa_layout(kwargs), cd_map(kwargs)["f16"]
    lrows = [
        ("S_PITCH", lay["s_pitch"], "one N plane of S^T: [latch][Bc key beats][m_old]"),
        ("P_PITCH", lay["p_pitch"], "one N half of P8: [Bc/4 B blocks][rowsum][corr*l_old]"),
        ("D16_TLBOUND0", f16["b0"], "QK's D32 walk at FP16: chunks per output block"),
        ("D16_TLSTRIDE1", f16["ts"][1], "                     the next N block = plane"),
        ("D16_TLSTRIDE2", f16["ts"][2], "                     the next M block = 16 keys"),
    ]
    n = max(len(name) for name, _, _ in lrows)
    w = max(len(str(v)) for _, v, _ in lrows)
    lbody = "\n".join(
        "#define %-*s %*du  // %s" % (n, name, w, value, note)
        for name, value, note in lrows
    )
    return (
        "// ---- TILE GEOMETRY -------------------------------------------------------"
        "---\n" + body + "\n" + lbody + "\n"
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
    # streamer c32 and d32 settings
    # -----------------------------------------------------------
    # C and D share ONE INT32 map (cd_map): the 16 channels of the port are one contiguous
    # 128 B transaction, and the temporal loops place output block (m, n) at
    # m*N*1 KiB + n*1 KiB, row-major inside. That is PV's O^T, read back through C and
    # written through D in place. QK's D writes FP16 score planes instead; the kernel
    # switches D's bound 0 and strides 1 and 2 to the D16_* walk for it.
    #
    #   spatial 0 : one channel                = 8 B
    #   spatial 1 : the next 4 channels        = 32 B    -> 16 channels = 128 B contiguous
    #   temporal 0: serial chunk               = 128 B   2 INT32 rows (4 FP16 rows)
    #   temporal 1: the next N block           = 1 KiB
    #   temporal 2: the next M block           = N KiB
    #
    # BOTH spatial strides must be emitted: the port declares two, so a 1-element array
    # feeds the second from off the end of the caller's stack and 12 of the 16 channels
    # address garbage.
    cdm = cd_map(kwargs)
    for port in ("C", "D32"):
        data_str += [format_scalar_definition("int32_t", port + "slstride0", cdm["sl"][0])]
        data_str += [format_scalar_definition("int32_t", port + "slstride1", cdm["sl"][1])]
        data_str += [format_scalar_definition("int32_t", port + "tlbound0", cdm["i32"]["b0"])]
        data_str += [
            format_scalar_definition("int32_t", port + "tlstride0", cdm["i32"]["ts"][0])
        ]
        data_str += [format_scalar_definition("int32_t", port + "tlbound1", kwargs["N"])]
        data_str += [
            format_scalar_definition("int32_t", port + "tlstride1", cdm["i32"]["ts"][1])
        ]
        data_str += [format_scalar_definition("int32_t", port + "tlbound2", kwargs["M"])]
        data_str += [
            format_scalar_definition("int32_t", port + "tlstride2", cdm["i32"]["ts"][2])
        ]
        data_str += [format_scalar_definition("int32_t", port + "tlbound3", 1)]
        data_str += [format_scalar_definition("int32_t", port + "tlstride3", 0)]

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
