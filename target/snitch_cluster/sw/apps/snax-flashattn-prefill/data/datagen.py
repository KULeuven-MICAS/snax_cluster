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


def attention_qshift(d):
    """Smallest right shift on the INT8 operands that keeps every score inside FP16.

    A score is a sum of d products of two shifted INT8s, so |S| <= (128>>q)^2 * d.
    Int32ToFp16 saturates past 65504, exp(inf - inf) is NaN, and the softmax invariant
    then fails on exactly the overflowing rows. Real attention scales by 1/sqrt(d) for the
    same reason; bounding the inputs is cheaper here. Deriving q from d lets the head
    dimension move without reintroducing the overflow.
    """
    for q in range(8):
        if (128 >> q) ** 2 * d <= 65504:
            return q
    raise ValueError("no INT8 shift keeps d=%d scores inside FP16" % d)


def emit_attention_golden(A, B, **kwargs):
    """A float model of the softmax over the same tile, for the kernel to compare against.

    The exact invariants in the kernel pin one element per query row -- the row maximum,
    which subtracts to zero and exponentiates to 1.0. They say nothing about the other
    Bc-1 elements. This does: it is the same scores, the same maximum and the same
    exponential computed in float, so a scale error in exp, a truncated accumulator or a
    lane that reduces the wrong operand all move a value that is checked here.

    The model follows the hardware's own sequence, at the precision the hardware uses:

        S      exact INT32 out of the mesh, converted to FP16 on the D32 port
        m      max over KEYS, per query row
        P      exp(S16 - m), FP32 internally, stored FP16
        rowsum sum over KEYS in the FP32 accumulator, narrowed to FP16

    S^T arrives as [Bc][Br]: one beat per key, one query row per lane. That is what lets
    the reduce run along beats, and it is the order the values are emitted in here.

    P is sampled rather than emitted whole -- one beat in PGOLD_STRIDE -- because the
    kernel reads the golden out of DRAM. rowsum is checked in full and integrates every
    key, so a wrong P between two samples still moves a checked number.

    P is checked as INT8, not FP16, because the epilogue pass quantises the tile in the
    same sweep that produces it -- there is no FP16 P in memory to compare against. The
    kernel's quantiser runs at inv_scale = 1.0, so q = round_rne(P) over P in (0,1]: the
    check pins the subtract, the exponential and the quantiser's rounding threshold. A
    lane whose P sits within 0.01 of the .5 boundary is emitted as -1 (don't care), since
    one ULP of LUT error there legitimately flips the result.
    """
    meshRow, tileSize, meshCol = _mesh(kwargs)
    M, N, K = kwargs["M"], kwargs["N"], kwargs["K"]
    Bc, Br = M * meshRow, N * meshCol

    # No zero-point and no bias: the kernel programs gen_subtraction_config(0, 0) and
    # zeroes C, so the mesh computes a plain product.
    D = block_gemm_golden_model(
        M, K, N, meshRow, tileSize, meshCol, A, B, 0, 0,
        np.zeros(M * N * meshRow * meshCol, dtype=np.int64),
    )

    # Block layout [M][N][meshRow][meshCol] -> [key][query], which is how the writer
    # lays the tile down and how the SIMD core reads it back.
    S = np.asarray(D).reshape(M, N, meshRow, meshCol)
    S = S.transpose(0, 2, 1, 3).reshape(Bc, Br)

    S16 = S.astype(np.float16)
    m = S16.max(axis=0)
    P = np.exp(S16.astype(np.float32) - m.astype(np.float32)).astype(np.float16)
    rowsum = P.astype(np.float32).sum(axis=0).astype(np.float16)

    stride = max(1, Bc // 16)
    beats = list(range(0, Bc, stride))

    # The kernel quantises P with inv_scale = 1.0 (SIMD_F32_ONE), clamp-then-round, then a
    # symmetric saturate -- the same order as Fp16ToInt8PE.
    def quantise(x):
        q = np.clip(np.rint(np.clip(x, -128.0, 128.0)), -127, 127).astype(np.int16)
        return np.where(np.abs(x - np.rint(x)) > 0.49, -1, q).astype(np.int16)

    Pfull = P.astype(np.float32)
    p8 = quantise(Pfull[beats].reshape(-1))
    # Per-query count of ones over the WHOLE tile. The sampled compare above pins values at
    # known places; this pins the tile globally -- a dropped beat, a tail in the wrong slot
    # or a shifted write moves a count even where every sampled beat still reads 0. Rows with
    # any boundary lane are excluded (-1), since their count is not well defined.
    p8all = quantise(Pfull.reshape(-1)).reshape(Pfull.shape)
    ones = (p8all == 1).sum(axis=0).astype(np.int16)
    ones = np.where((p8all < 0).any(axis=0), -1, ones).astype(np.int16)

    # ---- O, the second matmul's accumulator -----------------------------------
    #
    # oacc32 is NKV passes of V^T.P8^T added into themselves through C. Every KV tile is
    # fed the same K, so the running max stops moving after tile 0 and corr = exp(0) = 1
    # from then on: the online update degenerates to a plain sum of NKV identical P.V
    # products, which makes this the TRUE O for this data and not merely its shape.
    #
    # Built by the same generator as the matmul itself, with M and K swapped the way
    # emit_shape2_section() swaps them, so the layout this assumes and the layout the
    # descriptors walk come from one place. A is handed over as the raw buffer: shape 2
    # reinterprets shape 1's bytes as V^T, and the golden must reinterpret them the same
    # way rather than from some separately-derived idea of what V is.
    #
    # P8 goes in WITHOUT the don't-care substitution -- an O element sums 512 of them, so
    # a -1 would poison the sum. Instead the whole golden is marked invalid if any lane
    # sits near the rounding boundary, and the kernel skips the check and says so.
    p8_exact = np.clip(np.rint(np.clip(Pfull, -128.0, 128.0)), -127, 127)
    o_valid = 0 if (p8all < 0).any() else 1
    s2_m = (kwargs["K"] * tileSize) // meshRow      # d / meshRow
    s2_k = (M * meshRow) // tileSize                # Bc / tileSize
    o32 = block_gemm_golden_model(
        s2_m, s2_k, N, meshRow, tileSize, meshCol,
        A, p8_exact.reshape(-1).astype(np.int16), 0, 0,
        np.zeros(s2_m * N * meshRow * meshCol, dtype=np.int64),
    )
    o32 = np.asarray(o32, dtype=np.int64) * int(kwargs["NKV"])

    # ---- and now put it where the port actually writes it ----------------------
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
    o_mem = np.zeros(o32.size, dtype=np.int64)
    o_can = o32.reshape(s2_m, N, meshRow, meshCol)
    seen = np.zeros(o32.size, dtype=bool)
    for (mm, nn, r, c), b in scatter(output_data_width, s2_m, ts2, 4).items():
        w = b // 4
        assert b % 4 == 0 and not seen[w], "D32 INT32 address map is not a bijection"
        seen[w] = True
        o_mem[w] = o_can[mm, nn, r, c]
    assert seen.all(), "D32 INT32 address map does not cover the output"
    o32 = o_mem.astype(np.int32)

    def bits(x):
        return np.ascontiguousarray(x, dtype=np.float16).view(np.uint16).reshape(-1)

    return "\n".join([
        "// ---- ATTENTION GOLDEN ------------------------------------------------------",
        "// FP16 bit patterns, compared by ULP distance: the check core has no FPU.",
        "#define PGOLD_STRIDE %d  // every Nth beat of P is checked" % stride,
        "#define PGOLD_NBEATS %d" % len(beats),
        "",
        format_vector_definition("uint16_t", "m_golden", bits(m)),
        "",
        format_vector_definition("uint16_t", "rowsum_golden", bits(rowsum)),
        "",
        format_vector_definition("uint16_t", "p16_golden", bits(P[beats].reshape(-1))),
        "",
        "// P as the quantiser emits it: round_rne(P) at inv_scale 1.0, -1 = don't care",
        format_vector_definition("int16_t", "p8_golden", p8),
        "",
        "// P^T == 1 count per query row over the whole tile, -1 = don't care",
        format_vector_definition("int16_t", "p8ones_golden", ones),
        "",
        "// O = NKV * (V^T . P8^T), in the [M][N][meshRow][meshCol] order the D32 port",
        "// writes. 0 here means a P lane sat on the rounding boundary and the sum is not",
        "// reproducible; the kernel then skips the compare rather than reporting noise.",
        "#define O_GOLDEN_VALID %d" % o_valid,
        format_vector_definition("int32_t", "o32_golden", o32),
    ])


def emit_geometry_section(**kwargs):
    """The tile geometry as compile-time constants, derived from the shape and the mesh.

    The kernel takes BR/BC/DHEAD/NKV/QSHIFT from here rather than defining its own, so it
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
        ("NQ", kwargs["NQ"], "query tiles sharing one K/V pass -- intensity is NQ*Br"),
        (
            "QSHIFT",
            attention_qshift(d),
            "operand bound, ALREADY APPLIED to A and B below",
        ),
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

    A = np.random.randint(
        MIN, MAX, size=(kwargs["M"], kwargs["K"], meshRow, tileSize)
    ).reshape(-1)
    B = np.random.randint(
        MIN, MAX, size=(kwargs["K"], kwargs["N"], tileSize, meshCol)
    ).reshape(-1)

    # Bound the operands so the scores stay inside FP16 -- see attention_qshift(). Doing
    # it here rather than in the kernel keeps a shift over every element of A and B off
    # the DM core at run time; numpy's arithmetic shift on int8 is what `>>=` on int8_t
    # is in C, so the data is what the kernel would have produced itself.
    qshift = attention_qshift(kwargs["K"] * tileSize)
    A = A.astype(np.int8) >> qshift
    B = B.astype(np.int8) >> qshift

    # 64-BYTE ALIGNED, because these are xDMA sources. The iDMA copies bytes at any
    # alignment, but the xDMA reader issues eight 8-byte channels per beat and needs
    # its base aligned to the beat. Unaligned, it still moves the right number of
    # bytes at the right rate and only the DATA is wrong -- which showed up here as
    # m, P8 and rowsum all bit-exact (they come from K, loaded by the iDMA) while
    # 2866 of 4096 O elements were wrong (O comes from V, loaded by the xDMA).
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

    return data_str, A, B


def emit_versacore_data(**kwargs):
    data_str, A, B = emit_matmul_data(**kwargs)

    # No rescale epilogue and no raw-matmul golden. VersaCore has no rescale unit and this
    # cluster gives its write path no rescale extension, so there are no zero-point,
    # shift/multiplier or rounding CSRs to feed. The kernel checks the SOFTMAX, against
    # emit_attention_golden(), rather than the matmul, so a D32 vector would be dead
    # weight in data.h.

    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_A", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_B", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_C", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_D32", 0)]

    data_str += [emit_attention_golden(A, B, **kwargs)]

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

    shape2 = {**merged_config, "M": m2, "N": n2, "K": k2}
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
