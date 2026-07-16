#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data generator for the RUNTIME-PRECISION xDMA RoPE test. Emits ONE row of N
# values, snapped onto BOTH the FP16 grid and the FP8 (e5m2) grid, together with
# the per-pair-duplicated cos_full / sin_signed tables and the matching RoPE
# goldens. The SAME xDMA netlist (StreamElementwiseRt) runs the whole 3-pass
# chain at either precision, chosen at runtime by the `fmt` CSR field -- so the
# host test feeds the FP16 dataset with fmt=FP16 and the FP8 dataset with
# fmt=FP8 and checks each against its own grid-faithful golden.
#
# RoPE uses the INTERLEAVED adjacent-pair convention (Meta/complex-rotation):
# pair k = (x[2k], x[2k+1]) rotates by angle theta_k.
#   out[2k]   = x[2k]*cos_k - x[2k+1]*sin_k
#   out[2k+1] = x[2k]*sin_k + x[2k+1]*cos_k
# Decomposed into 3 StreamElementwiseRt passes over per-pair-duplicated tables:
#   P1 tmp1 = x     (.) cos_full      cos_full   = [c0,c0,c1,c1,...]
#   P2 tmp2 = xswap (.) sin_signed    sin_signed = [-s0,+s0,-s1,+s1,...] (sign in)
#   P3 out  = tmp1  (+) tmp2          xswap      = [x1,x0,x3,x2,...] (rotate_half)
#
# The goldens MIRROR the hardware datapath EXACTLY (FP32-internal, values snapped
# onto the fmt grid at each stage boundary): every pass widens its fmt operands
# to FP32, computes in FP32, and narrows the result back to fmt. cos_full,
# sin_signed and x are themselves snapped onto the fmt grid first (that is what
# the HW loads and multiplies), and the rotate_half swap operates on the fmt
# code words (2-byte halfword swap at FP16, 1-byte byte swap at FP8), matching
# the iDMA staging in the .c.

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402

FP16_PER_BEAT = 32
FP8_PER_BEAT = 64
RNG_SEED = 320
ROPE_BASE = 10000.0
ROPE_POS = 1


# ---- FP8 e5m2 codec (1 sign, 5 exp bias 15, 2 mant), RNE + subnormal flush-to-zero -------------
def f32_to_e5m2_bits(x):
    x = np.asarray(x, dtype=np.float32)
    a = np.abs(x).astype(np.float64)
    sign = np.where(np.signbit(x), 0x80, 0).astype(np.int32)
    nz = a > 0.0
    e = np.floor(np.log2(np.where(nz, a, 1.0))).astype(np.int32)
    exp = e + 15
    frac = np.where(nz, a / np.power(2.0, e.astype(np.float64)) - 1.0, 0.0)
    mant = np.rint(frac * 4.0).astype(np.int32)           # RNE
    carry = mant == 4
    mant = np.where(carry, 0, mant)
    exp = np.where(carry, exp + 1, exp)
    under = (~nz) | (exp < 1)
    over = exp >= 0x1F
    bits = sign | (exp << 2) | mant
    bits = np.where(over, sign | 0x7B, bits)              # clamp to max finite (57344)
    bits = np.where(under, 0, bits)                       # FTZ (matches narrowRt)
    return bits.astype(np.uint8)


def e5m2_bits_to_f32(b):
    b = np.asarray(b, dtype=np.uint8).astype(np.int32)
    sign = np.where((b & 0x80) != 0, -1.0, 1.0)
    exp = (b >> 2) & 0x1F
    mant = b & 0x3
    val = np.where(exp == 0,
                   sign * mant * (2.0 ** -16),
                   sign * (4 + mant) * np.power(2.0, (exp - 17).astype(np.float64)))
    return val.astype(np.float32)


def rope_angles(n):
    # theta_k for a head_dim = N row at position ROPE_POS (FP64, unsnapped; each
    # format snaps c/s onto its own grid below, matching what the HW loads).
    half = n // 2
    inv_freq = ROPE_BASE ** (-(np.arange(half).astype(np.float64)) / half)
    ang = ROPE_POS * inv_freq
    return np.cos(ang), np.sin(ang)


def rope_chain_fp16(xf, c, s, n):
    x16 = xf.astype(np.float16)
    cos_full = np.repeat(c.astype(np.float16), 2)         # [c0,c0,c1,c1,...]
    sin_signed = np.empty(n, np.float32)
    sin_signed[0::2] = -s                                 # even lanes: -sin
    sin_signed[1::2] = +s                                 # odd  lanes: +sin
    sin_signed = sin_signed.astype(np.float16)
    xu = x16.view(np.uint16)                              # rotate_half (halfword swap):
    xswap = np.empty(n, np.uint16)                        # xswap[2k]=x[2k+1], [2k+1]=x[2k]
    xswap[0::2] = xu[1::2]
    xswap[1::2] = xu[0::2]
    xswap = xswap.view(np.float16)
    tmp1 = (x16.astype(np.float32) * cos_full.astype(np.float32)).astype(np.float16)      # P1 MUL
    tmp2 = (xswap.astype(np.float32) * sin_signed.astype(np.float32)).astype(np.float16)  # P2 MUL
    golden = (tmp1.astype(np.float32) + tmp2.astype(np.float32)).astype(np.float16)       # P3 ADD
    return (x16.view(np.uint16), cos_full.view(np.uint16),
            sin_signed.view(np.uint16), golden.view(np.uint16))


def rope_chain_fp8(xf, c, s, n):
    x8 = f32_to_e5m2_bits(xf)
    x8_f32 = e5m2_bits_to_f32(x8)
    cos_full = np.repeat(f32_to_e5m2_bits(c.astype(np.float32)), 2)   # [c0,c0,c1,c1,...]
    cos_full_f32 = e5m2_bits_to_f32(cos_full)
    sin_f = np.empty(n, np.float32)
    sin_f[0::2] = -s                                      # even lanes: -sin
    sin_f[1::2] = +s                                      # odd  lanes: +sin
    sin_signed = f32_to_e5m2_bits(sin_f)
    sin_signed_f32 = e5m2_bits_to_f32(sin_signed)
    xswap = np.empty(n, np.uint8)                         # rotate_half (byte swap):
    xswap[0::2] = x8[1::2]                                # xswap[2k]=x[2k+1], [2k+1]=x[2k]
    xswap[1::2] = x8[0::2]
    xswap_f32 = e5m2_bits_to_f32(xswap)
    tmp1 = f32_to_e5m2_bits(x8_f32 * cos_full_f32)                    # P1 MUL (fmt-narrowed)
    tmp1_f32 = e5m2_bits_to_f32(tmp1)
    tmp2 = f32_to_e5m2_bits(xswap_f32 * sin_signed_f32)              # P2 MUL (fmt-narrowed)
    tmp2_f32 = e5m2_bits_to_f32(tmp2)
    golden = f32_to_e5m2_bits(tmp1_f32 + tmp2_f32)                   # P3 ADD (fmt-narrowed)
    return x8, cos_full, sin_signed, golden


def emit_header_file(**kwargs):
    n = int(kwargs["N"])
    if n <= 0 or n % FP8_PER_BEAT != 0:
        raise ValueError(f"N must be a positive multiple of {FP8_PER_BEAT}, got {n}")

    rng = np.random.default_rng(RNG_SEED)
    xf = rng.uniform(-4.0, 4.0, size=n).astype(np.float32)
    c, s = rope_angles(n)

    x16u, cos16u, sin16u, g16u = rope_chain_fp16(xf, c, s, n)
    x8u, cos8u, sin8u, g8u = rope_chain_fp8(xf, c, s, n)

    emit = ["#include <stdint.h>"]
    emit += [format_scalar_definition("uint32_t", "rope_n", n)]
    emit += [format_scalar_definition("uint32_t", "rope_beats_fp16", n // FP16_PER_BEAT)]
    emit += [format_scalar_definition("uint32_t", "rope_beats_fp8", n // FP8_PER_BEAT)]
    # FP16 dataset
    emit += [format_vector_definition("uint16_t", "rope_x_fp16", x16u,
                                      alignment=64, hex_bits=16, cast_hex=True)]
    emit += [format_vector_definition("uint16_t", "rope_cos_fp16", cos16u,
                                      alignment=64, hex_bits=16, cast_hex=True)]
    emit += [format_vector_definition("uint16_t", "rope_sin_fp16", sin16u,
                                      alignment=64, hex_bits=16, cast_hex=True)]
    emit += [format_vector_definition("uint16_t", "rope_golden_fp16", g16u,
                                      alignment=64, hex_bits=16, cast_hex=True)]
    # FP8 dataset
    emit += [format_vector_definition("uint8_t", "rope_x_fp8", x8u,
                                      alignment=64, hex_bits=8, cast_hex=True)]
    emit += [format_vector_definition("uint8_t", "rope_cos_fp8", cos8u,
                                      alignment=64, hex_bits=8, cast_hex=True)]
    emit += [format_vector_definition("uint8_t", "rope_sin_fp8", sin8u,
                                      alignment=64, hex_bits=8, cast_hex=True)]
    emit += [format_vector_definition("uint8_t", "rope_golden_fp8", g8u,
                                      alignment=64, hex_bits=8, cast_hex=True)]
    return "\n\n".join(emit)


def main():
    parser = argparse.ArgumentParser(description="Data for the runtime-precision xDMA RoPE kernel")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
