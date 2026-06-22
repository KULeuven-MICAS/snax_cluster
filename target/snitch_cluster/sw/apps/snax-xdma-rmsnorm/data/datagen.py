#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data generator for the xDMA FP16 rmsnorm test. Emits one FP16 row of length N
# (a power of two, a multiple of 32) and the FP16 golden rmsnorm = x * rsqrt(mean(x^2)).
# N is a power of two so the DM core can form mean = sum(x^2)/N by an integer
# exponent-field subtraction (no FPU); eps = 0 (the inputs keep mean(x^2) well away
# from 0, so eps is unnecessary for this benchmark).

import argparse
import math
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402

FP16_PER_BEAT = 32
RNG_SEED = 320


def emit_header_file(**kwargs):
    n = int(kwargs["N"])
    if n <= 0 or n % FP16_PER_BEAT != 0:
        raise ValueError(f"N must be a positive multiple of {FP16_PER_BEAT}, got {n}")
    if n & (n - 1) != 0:
        raise ValueError(f"N must be a power of two (for the /N exponent trick), got {n}")
    beats = n // FP16_PER_BEAT
    log2n = int(round(math.log2(n)))

    rng = np.random.default_rng(RNG_SEED)
    xf = rng.uniform(-4.0, 4.0, size=n).astype(np.float32)
    x16 = xf.astype(np.float16)            # snap inputs onto the FP16 grid
    xf32 = x16.astype(np.float32)

    mean = np.float64((xf32.astype(np.float64) ** 2).sum()) / n
    inv_rms = np.float64(1.0) / np.sqrt(mean)            # eps = 0
    out16 = (xf32.astype(np.float64) * inv_rms).astype(np.float16)  # golden rmsnorm -> FP16

    # INT8 quantize golden for the fused Fp16ToInt8 stage: q = sat127(rne(fp32(out16)*inv_scale)).
    # Mirrors the HW PE (fp32 product, clamp to +/-128 before the RNE round, symmetric saturate).
    inv_scale = np.float32(64.0)
    prod = np.clip(out16.astype(np.float32) * inv_scale, np.float32(-128.0), np.float32(128.0))
    q_i8 = np.clip(np.rint(prod.astype(np.float64)), -127, 127).astype(np.int8)

    # Host-computed scalar rsqrt (1/sqrt(mean)): no longer an xDMA op. T1's SUMSQ emits Σx² narrowed to
    # FP16; the DM core forms mean = Σx²/N via the exact exponent-field subtract (N=2^log2n), and the host
    # computes 1/sqrt(mean). Mirror exactly so the precomputed inverse matches the runtime reduce.
    ssq_fp16 = np.float16((xf32 ** 2).sum(dtype=np.float32))    # FP32 accumulate -> FP16 trailing beat
    mean_fp32 = ssq_fp16.astype(np.float32) / np.float32(n)     # = fp32(Σx²)/2^log2n (exact)
    inv_rms_fp32 = np.float32(1.0) / np.float32(np.sqrt(mean_fp32))

    x_u16 = x16.view(np.uint16)
    g_u16 = out16.view(np.uint16)

    emit = ["#include <stdint.h>"]
    emit += [format_scalar_definition("uint32_t", "rmsnorm_n", n)]
    emit += [format_scalar_definition("uint32_t", "rmsnorm_beats", beats)]
    emit += [format_scalar_definition("uint32_t", "rmsnorm_log2n", log2n)]
    # host-provided scalar rsqrt: FP16 Σx² (runtime-reduce check) + FP32 bits of 1/sqrt(mean) (map operand a)
    emit += [format_scalar_definition("uint32_t", "rmsnorm_ssq_golden", int(ssq_fp16.view(np.uint16)))]
    emit += [format_scalar_definition("uint32_t", "rmsnorm_inv_rms", int(inv_rms_fp32.view(np.uint32)))]
    emit += [
        format_vector_definition(
            "uint16_t", "rmsnorm_input", x_u16,
            alignment=64, hex_bits=16, cast_hex=True,
        )
    ]
    emit += [
        format_vector_definition(
            "uint16_t", "rmsnorm_golden", g_u16,
            alignment=64, hex_bits=16, cast_hex=True,
        )
    ]
    emit += [format_scalar_definition("uint32_t", "rmsnorm_inv_scale", int(inv_scale.view(np.uint32)))]
    emit += [format_vector_definition("int8_t", "rmsnorm_golden_i8", q_i8, alignment=64)]
    return "\n\n".join(emit)


def main():
    parser = argparse.ArgumentParser(description="Generating data for the xDMA rmsnorm kernel")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
