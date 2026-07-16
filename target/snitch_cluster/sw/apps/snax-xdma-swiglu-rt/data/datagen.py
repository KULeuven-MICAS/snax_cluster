#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data generator for the RUNTIME-PRECISION xDMA SwiGLU test: out = silu(gate) (.) up.
# Emits TWO rows of N values (`gate`, `up`), snapped onto BOTH the FP16 grid and the
# FP8 (e5m2) grid, plus the matching swiglu goldens. The SAME xDMA netlists
# (StreamMapRt = idx 6 for silu, StreamElementwiseRt = idx 7 for the multiply) run the
# whole chain at either precision, chosen at runtime by the `fmt` CSR field -- so the
# host test feeds the FP16 dataset with fmt=FP16 and the FP8 dataset with fmt=FP8 and
# checks each against its own grid-faithful golden.
#
# The goldens MIRROR the hardware datapath exactly (FP32-internal, values snapped at
# each stage boundary):
#   T1 StreamMapRt(SILU) : widenRt(gate_grid)->FP32, silu(x) in FP32, narrowRt to fmt.
#   T2 StreamElementwiseRt(MUL, operandCount=2) : FP32 product of the two fmt operands
#      (silu(gate) and up), narrowRt to fmt.
# The activation itself is evaluated with the true (FP64) silu; the small LUT error of
# the HW FpSilu is absorbed by the ULP tolerance in the host check.

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


def swiglu_chain_fp16(gate_f, up_f):
    g16 = gate_f.astype(np.float16)                       # gate on the FP16 grid
    u16 = up_f.astype(np.float16)                         # up   on the FP16 grid
    g32 = g16.astype(np.float32)                          # widenRt: grid -> FP32 (exact)
    # T1 StreamMapRt(SILU): silu FP32-internal, narrowed to fmt.
    sg = (g32.astype(np.float64) / (1.0 + np.exp(-g32.astype(np.float64))))
    sg16 = sg.astype(np.float16)                          # narrowRt -> FP16
    # T2 StreamElementwiseRt(MUL): FP32 product of the two FP16 operands, narrowed to fmt.
    out16 = (sg16.astype(np.float32) * u16.astype(np.float32)).astype(np.float16)
    return g16.view(np.uint16), u16.view(np.uint16), out16.view(np.uint16)


def swiglu_chain_fp8(gate_f, up_f):
    g8 = f32_to_e5m2_bits(gate_f)                         # gate on the FP8 grid
    u8 = f32_to_e5m2_bits(up_f)                           # up   on the FP8 grid
    g32 = e5m2_bits_to_f32(g8)                            # widenRt: grid -> FP32
    # T1 StreamMapRt(SILU): silu FP32-internal, narrowed to fmt.
    sg = (g32.astype(np.float64) / (1.0 + np.exp(-g32.astype(np.float64))))
    sg8 = f32_to_e5m2_bits(sg.astype(np.float32))         # narrowRt -> FP8
    sg8_f32 = e5m2_bits_to_f32(sg8)
    u8_f32 = e5m2_bits_to_f32(u8)
    # T2 StreamElementwiseRt(MUL): FP32 product of the two FP8 operands, narrowed to fmt.
    out8 = f32_to_e5m2_bits(sg8_f32 * u8_f32)
    return g8, u8, out8


def emit_header_file(**kwargs):
    n = int(kwargs["N"])
    if n <= 0 or n % FP8_PER_BEAT != 0:
        raise ValueError(f"N must be a positive multiple of {FP8_PER_BEAT}, got {n}")

    # Same rng feeds both grids: gate then up, each uniform(-4,4).
    rng = np.random.default_rng(RNG_SEED)
    gate = rng.uniform(-4.0, 4.0, size=n).astype(np.float32)
    up = rng.uniform(-4.0, 4.0, size=n).astype(np.float32)

    g16u, u16u, out16u = swiglu_chain_fp16(gate, up)
    g8u, u8u, out8u = swiglu_chain_fp8(gate, up)

    emit = ["#include <stdint.h>"]
    emit += [format_scalar_definition("uint32_t", "swiglu_n", n)]
    emit += [format_scalar_definition("uint32_t", "swiglu_beats_fp16", n // FP16_PER_BEAT)]
    emit += [format_scalar_definition("uint32_t", "swiglu_beats_fp8", n // FP8_PER_BEAT)]
    # FP16 dataset
    emit += [format_vector_definition("uint16_t", "swiglu_gate_fp16", g16u,
                                      alignment=64, hex_bits=16, cast_hex=True)]
    emit += [format_vector_definition("uint16_t", "swiglu_up_fp16", u16u,
                                      alignment=64, hex_bits=16, cast_hex=True)]
    emit += [format_vector_definition("uint16_t", "swiglu_golden_fp16", out16u,
                                      alignment=64, hex_bits=16, cast_hex=True)]
    # FP8 dataset
    emit += [format_vector_definition("uint8_t", "swiglu_gate_fp8", g8u,
                                      alignment=64, hex_bits=8, cast_hex=True)]
    emit += [format_vector_definition("uint8_t", "swiglu_up_fp8", u8u,
                                      alignment=64, hex_bits=8, cast_hex=True)]
    emit += [format_vector_definition("uint8_t", "swiglu_golden_fp8", out8u,
                                      alignment=64, hex_bits=8, cast_hex=True)]
    return "\n\n".join(emit)


def main():
    parser = argparse.ArgumentParser(description="Data for the runtime-precision xDMA swiglu kernel")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
