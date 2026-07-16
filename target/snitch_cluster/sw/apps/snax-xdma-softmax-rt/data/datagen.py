#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data generator for the RUNTIME-PRECISION xDMA softmax test. Emits ONE row of N
# values, snapped onto BOTH the FP16 grid and the FP8 (e5m2) grid, plus the
# matching softmax goldens. The SAME xDMA netlists (StreamMapRt / StreamReduceRt)
# run the whole chain at either precision, chosen at runtime by the `fmt` CSR
# field -- so the host test feeds the FP16 dataset with fmt=FP16 and the FP8
# dataset with fmt=FP8 and checks each against its own grid-faithful golden.
#
# The goldens MIRROR the hardware datapath exactly (FP32-internal, values snapped
# at each stage boundary): reduce-max -> map exp(x-m) narrowed to fmt -> reduce
# ADD (FP32-out, so the host reciprocal is accurate) -> map (a=1/s) narrowed to
# fmt. inv_sum is the host reciprocal of the FP32 sum (the reduce runs with the
# fp32out CSR bit so the trailing scalar beat carries the true FP32 sum).

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


def softmax_chain_fp16(xf):
    x16 = xf.astype(np.float16)
    x32 = x16.astype(np.float32)
    m = np.float32(x32.max())
    e = np.exp((x32 - m).astype(np.float64))
    e16 = e.astype(np.float16)                            # map exp output (fmt-narrowed)
    e16_f32 = e16.astype(np.float32)
    s = np.float32(e16_f32.sum(dtype=np.float32))         # reduce ADD, FP32-out
    inv = np.float32(1.0) / s
    y16 = (e16_f32 * inv).astype(np.float16)              # map normalize output
    return x16.view(np.uint16), y16.view(np.uint16), e16.view(np.uint16), s, inv


def softmax_chain_fp8(xf):
    x8 = f32_to_e5m2_bits(xf)
    x32 = e5m2_bits_to_f32(x8)
    m = np.float32(x32.max())
    e = np.exp((x32 - m).astype(np.float64)).astype(np.float32)
    e8 = f32_to_e5m2_bits(e)                              # map exp output (fmt-narrowed)
    e8_f32 = e5m2_bits_to_f32(e8)
    s = np.float32(e8_f32.sum(dtype=np.float32))          # reduce ADD, FP32-out
    inv = np.float32(1.0) / s
    y8 = f32_to_e5m2_bits(e8_f32 * inv)                   # map normalize output
    return x8, y8, e8, s, inv


def emit_header_file(**kwargs):
    n = int(kwargs["N"])
    if n <= 0 or n % FP8_PER_BEAT != 0:
        raise ValueError(f"N must be a positive multiple of {FP8_PER_BEAT}, got {n}")

    rng = np.random.default_rng(RNG_SEED)
    xf = rng.uniform(-4.0, 4.0, size=n).astype(np.float32)

    x16u, g16u, _e16u, s16, inv16 = softmax_chain_fp16(xf)
    x8u, g8u, _e8u, s8, inv8 = softmax_chain_fp8(xf)

    emit = ["#include <stdint.h>"]
    emit += [format_scalar_definition("uint32_t", "softmax_n", n)]
    emit += [format_scalar_definition("uint32_t", "softmax_beats_fp16", n // FP16_PER_BEAT)]
    emit += [format_scalar_definition("uint32_t", "softmax_beats_fp8", n // FP8_PER_BEAT)]
    # host reciprocals (FP32 bits of 1/Σ) + FP32 Σ goldens (reduce runs fp32out)
    emit += [format_scalar_definition("uint32_t", "softmax_inv_sum_fp16", int(inv16.view(np.uint32)))]
    emit += [format_scalar_definition("uint32_t", "softmax_inv_sum_fp8", int(inv8.view(np.uint32)))]
    emit += [format_scalar_definition("uint32_t", "softmax_sum_fp16", int(np.float32(s16).view(np.uint32)))]
    emit += [format_scalar_definition("uint32_t", "softmax_sum_fp8", int(np.float32(s8).view(np.uint32)))]
    # FP16 dataset
    emit += [format_vector_definition("uint16_t", "softmax_input_fp16", x16u,
                                      alignment=64, hex_bits=16, cast_hex=True)]
    emit += [format_vector_definition("uint16_t", "softmax_golden_fp16", g16u,
                                      alignment=64, hex_bits=16, cast_hex=True)]
    # FP8 dataset
    emit += [format_vector_definition("uint8_t", "softmax_input_fp8", x8u,
                                      alignment=64, hex_bits=8, cast_hex=True)]
    emit += [format_vector_definition("uint8_t", "softmax_golden_fp8", g8u,
                                      alignment=64, hex_bits=8, cast_hex=True)]
    return "\n\n".join(emit)


def main():
    parser = argparse.ArgumentParser(description="Data for the runtime-precision xDMA softmax kernel")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
