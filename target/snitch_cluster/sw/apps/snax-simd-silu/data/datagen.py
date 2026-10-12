#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data generator for the SIMD FP16 SiLU test. Emits one FP16 row of length N and the FP16 golden
# silu(x) = x * sigmoid(x) = x / (1 + e^-x). Inputs are snapped onto the FP16 grid first so the
# golden reflects exactly what the hardware consumes.
#
# The SIMD beat holds simd_fp16_lanes(hw) FP16 lanes, read from the hardware cfg, so the row is
# N / lanes beats. The fused quantise packs 2 beats into 1, so N must be a whole number of beat
# pairs.

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402
from snax_utils import simd_fp16_lanes  # noqa E402

RNG_SEED = 320


def emit_header_file(lanes, **kwargs):
    n = int(kwargs["N"])
    if n <= 0 or n % (2 * lanes) != 0:
        raise ValueError(f"N must be a positive multiple of 2 beats ({2 * lanes}), got {n}")
    beats = n // lanes

    rng = np.random.default_rng(RNG_SEED)
    xf = rng.uniform(-8.0, 8.0, size=n).astype(np.float32)
    x16 = xf.astype(np.float16)            # snap inputs onto the FP16 grid
    xf32 = x16.astype(np.float32).astype(np.float64)

    sig = 1.0 / (1.0 + np.exp(-xf32))
    out16 = (xf32 * sig).astype(np.float16)  # golden silu -> FP16

    # INT8 quantize golden for the fused Fp16ToInt8 stage: q = sat127(rne(fp32(out16)*inv_scale)).
    # Mirrors the HW PE (fp32 product, clamp to +/-128 before the RNE round, symmetric saturate).
    inv_scale = np.float32(16.0)
    prod = np.clip(out16.astype(np.float32) * inv_scale, np.float32(-128.0), np.float32(128.0))
    q_i8 = np.clip(np.rint(prod.astype(np.float64)), -127, 127).astype(np.int8)

    x_u16 = x16.view(np.uint16)
    g_u16 = out16.view(np.uint16)

    emit = ["#include <stdint.h>"]
    emit += [format_scalar_definition("uint32_t", "silu_n", n)]
    emit += [format_scalar_definition("uint32_t", "silu_beats", beats)]
    emit += [
        format_vector_definition(
            "uint16_t", "silu_input", x_u16,
            alignment=64, hex_bits=16, cast_hex=True,
        )
    ]
    emit += [
        format_vector_definition(
            "uint16_t", "silu_golden", g_u16,
            alignment=64, hex_bits=16, cast_hex=True,
        )
    ]
    emit += [format_scalar_definition("uint32_t", "silu_inv_scale", int(inv_scale.view(np.uint32)))]
    emit += [format_vector_definition("int8_t", "silu_golden_i8", q_i8, alignment=64)]
    return "\n\n".join(emit)


def main():
    parser = argparse.ArgumentParser(description="Generating data for the SIMD silu kernel")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True,
                        help="cluster hardware cfg (sets the SIMD lane count)")
    args = parser.parse_args()
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    with args.hwcfg.open() as f:
        hw = hjson.loads(f.read())
    print(emit_header_file(simd_fp16_lanes(hw), **param))


if __name__ == "__main__":
    main()
