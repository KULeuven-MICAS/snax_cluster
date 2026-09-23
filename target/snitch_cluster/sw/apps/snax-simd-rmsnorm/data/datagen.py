#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data for the RMSNorm kernel: one [rows, cols] FP16 tile, emitted BOTH ways round.
#
# x    [rows, cols]  row-major -- one token per row, the layout the layer hands the kernel
# x_t  [cols, rows]  transposed -- one token per FP16 LANE, the layout that makes the
#                    reduction run along beats instead of across lanes
#
# Same numbers, same goldens, so the two paths in the app are directly comparable.
#
# TWO GOLDENS, because the device has two different rsqrts and they do not agree:
#
#   out_golden  uses the core's INTEGER sqrt_f16 + recip_f16 -- what the legacy path must
#               reproduce, checked tight. This file models those two routines bit for bit,
#               so if the C port ever drifts the app says so instead of silently measuring
#               a different kernel.
#   out_exact   uses the true 1/sqrt(mean) -- the reference the HARDWARE rsqrt is scored
#               against, and the one that shows the hardware is the MORE accurate of the two.

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402

FP16_PER_BEAT = 32
RNG_SEED = 0x12345


def recip_f16(s: int) -> int:
    """Bit-exact model of recip_f16() -- the core's integer FP16 reciprocal. One divu."""
    E = (s >> 10) & 0x1F
    M = 1024 + (s & 0x3FF)
    q = ((1 << 21) + (M >> 1)) // M
    if q >= 2048:
        exp_field, mant = 30 - E, 0
    else:
        exp_field, mant = 29 - E, q - 1024
    return ((exp_field << 10) | (mant & 0x3FF)) & 0xFFFF


def sqrt_f16(v: int) -> int:
    """Bit-exact model of sqrt_f16() -- five Newton iterations, five divu."""
    E = (v >> 10) & 0x1F
    if E == 0:
        return 0
    M = 1024 + (v & 0x3FF)
    e = E - 15
    if e & 1:
        sig, oe = 2 * M, (e - 1) >> 1
    else:
        sig, oe = M, e >> 1
    n = sig << 10
    x = 1448
    for _ in range(5):
        x = (x + n // x) >> 1
    return (((oe + 15) << 10) | ((x - 1024) & 0x3FF)) & 0xFFFF


def inv_rms_int(ssq_bits: int, log2d: int) -> int:
    """The core's whole scalar epilogue: mean by exponent subtract, then the rsqrt."""
    Es = (ssq_bits >> 10) & 0x1F
    mean = (((Es - log2d) << 10) | (ssq_bits & 0x3FF)) & 0xFFFF
    return recip_f16(sqrt_f16(mean))


def fp16_mono(h: int) -> int:
    mag = h & 0x7FFF
    return (0x8000 - mag) if (h & 0x8000) else (0x8000 + mag)


def emit_header_file(**kwargs):
    rows = int(kwargs["ROWS"])
    d = int(kwargs["D"])
    if d <= 0 or d % FP16_PER_BEAT != 0:
        raise ValueError(f"D must be a positive multiple of {FP16_PER_BEAT}, got {d}")
    if d & (d - 1) != 0:
        raise ValueError(
            f"D must be a power of two: both the core's mean and the hardware rsqrt's "
            f"a = 1/D are exponent-only, which is exact at 2^k and not otherwise. Got {d}.")
    if rows != FP16_PER_BEAT:
        raise ValueError(
            f"ROWS must be {FP16_PER_BEAT}: the transposed path puts one token per FP16 "
            f"lane, so a whole tile's per-token scalars are exactly one beat. Got {rows}.")
    beats = d // FP16_PER_BEAT
    log2d = int(d).bit_length() - 1

    rng = np.random.default_rng(RNG_SEED)
    x16 = rng.uniform(-4.0, 4.0, size=(rows, d)).astype(np.float32).astype(np.float16)
    xf32 = x16.astype(np.float32)

    # The reduce accumulates in FP32 and narrows the scalar to FP16 -- so does this.
    ssq16 = np.array([np.float16((xf32[r] ** 2).sum(dtype=np.float32)) for r in range(rows)],
                     dtype=np.float16)
    ssq_bits = ssq16.view(np.uint16)

    inv_int = np.array([inv_rms_int(int(b), log2d) for b in ssq_bits], dtype=np.uint16)
    inv_int_f = inv_int.view(np.float16).astype(np.float32)
    mean = ssq16.astype(np.float32) / np.float32(d)                    # exact: D = 2^k
    inv_exact = np.float16(np.float32(1.0) / np.sqrt(mean)).astype(np.float16)

    out_int = np.empty((rows, d), dtype=np.float16)
    out_exact = np.empty((rows, d), dtype=np.float16)
    for r in range(rows):
        out_int[r] = (xf32[r] * inv_int_f[r]).astype(np.float16)
        out_exact[r] = (xf32[r] * inv_exact[r].astype(np.float32)).astype(np.float16)

    # How far the core's FPU-less rsqrt sits from the true one, reported here so the app
    # does not have to rediscover it: this is the accuracy the hardware rsqrt REPLACES.
    worst_inv_ulp = max(abs(fp16_mono(int(a)) - fp16_mono(int(b)))
                        for a, b in zip(inv_int, inv_exact.view(np.uint16)))

    emit = ["#include <stdint.h>",
            f"// core integer rsqrt vs the true rsqrt, on inv_rms: worst {worst_inv_ulp} FP16 ULP"]
    emit += [format_scalar_definition("uint32_t", "rms_rows", rows)]
    emit += [format_scalar_definition("uint32_t", "rms_d", d)]
    emit += [format_scalar_definition("uint32_t", "rms_beats", beats)]
    emit += [format_scalar_definition("uint32_t", "rms_log2d", log2d)]
    emit += [format_scalar_definition("uint32_t", "rms_inv_rsqrt_ulp", worst_inv_ulp)]

    def vec(name, arr):
        return format_vector_definition("uint16_t", name, arr, alignment=64,
                                        hex_bits=16, cast_hex=True)

    emit += [vec("rms_input", x16.reshape(-1).view(np.uint16))]
    emit += [vec("rms_input_t", x16.T.copy().reshape(-1).view(np.uint16))]
    emit += [vec("rms_ssq_golden", ssq_bits)]
    emit += [vec("rms_inv_int_golden", inv_int)]
    emit += [vec("rms_out_golden", out_int.reshape(-1).view(np.uint16))]
    emit += [vec("rms_out_exact", out_exact.reshape(-1).view(np.uint16))]
    return "\n\n".join(emit)


def main():
    parser = argparse.ArgumentParser(description="Data for the SIMD FP16 RMSNorm kernel")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
