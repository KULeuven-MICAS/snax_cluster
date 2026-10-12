#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data generator for the SIMD FP16 RoPE test. Emits one FP16 row x of length N (a whole number of
# SIMD beats: simd_fp16_lanes(hw) lanes each, read from the hardware cfg), treated as a single
# head_dim row at position pos. RoPE uses the INTERLEAVED adjacent-pair
# convention
# (Meta/complex-rotation): pair k = (x[2k], x[2k+1]) rotates by angle theta_k.
#
#   out[2k]   = x[2k]*cos_k - x[2k+1]*sin_k
#   out[2k+1] = x[2k]*sin_k + x[2k+1]*cos_k
#
# On the SIMD block this is 3 StreamElementwise passes over per-pair-duplicated tables (no new HW):
#   P1 tmp1 = x     (.) cos_full      cos_full   = [c0,c0,c1,c1,...]
#   P2 tmp2 = xswap (.) sin_signed    sin_signed = [-s0,+s0,-s1,+s1,...]  (sign folded in)
#   P3 out  = tmp1  (+) tmp2          xswap      = [x1,x0,x3,x2,...] (adjacent halfword swap,
#   runtime)
# We emit cos_full and sin_signed (what the HW multiplies) and the golden. Inputs are snapped
# onto the
# FP16 grid first, with FP32-internal math, so the goldens reflect what the hardware consumes.

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
ROPE_BASE = 10000.0
ROPE_POS = 1


def emit_header_file(lanes, **kwargs):
    n = int(kwargs["N"])
    if n <= 0 or n % lanes != 0:
        raise ValueError(f"N must be a positive multiple of the {lanes} SIMD lanes, got {n}")
    beats = n // lanes
    half = n // 2

    # theta_k for a head_dim = N row at position ROPE_POS. FP16-snapped, FP32 math (matches HW
    # path).
    inv_freq = ROPE_BASE ** (-(np.arange(half).astype(np.float64)) / half)
    ang = ROPE_POS * inv_freq
    c = np.cos(ang).astype(np.float16).astype(np.float32)
    s = np.sin(ang).astype(np.float16).astype(np.float32)

    rng = np.random.default_rng(RNG_SEED)
    x = rng.uniform(-4.0, 4.0, size=n).astype(np.float16)

    # Per-pair-duplicated tables the HW multiplies lane-wise.
    cos_full = np.repeat(c, 2).astype(np.float16)        # [c0,c0,c1,c1,...]
    sin_signed = np.empty(n, np.float32)
    sin_signed[0::2] = -s                                 # even lanes: -sin
    sin_signed[1::2] = +s                                 # odd  lanes: +sin
    sin_signed = sin_signed.astype(np.float16)

    # Golden models the EXACT 3-pass HW arithmetic (so the on-cluster ULP gate verifies the
    # datapath):
    # each StreamElementwise pass widens its FP16 operands to FP32, computes in FP32, narrows to
    # FP16.
    xu = x.view(np.uint16)                                            # adjacent halfword swap of x:
    xswap = np.empty(n, np.uint16)
    # xswap[2k]=x[2k+1], xswap[2k+1]=x[2k]
    xswap[0::2] = xu[1::2]
    xswap[1::2] = xu[0::2]
    xswap = xswap.view(np.float16)
    tmp1 = (x.astype(np.float32) * cos_full.astype(np.float32)).astype(np.float16)        # P1 MUL
    tmp2 = (xswap.astype(np.float32) * sin_signed.astype(np.float32)).astype(np.float16)  # P2 MUL
    golden = (tmp1.astype(np.float32) + tmp2.astype(np.float32)).astype(np.float16)       # P3 ADD

    # Accuracy note vs an ideal FP32-fused RoPE (narrowed once). The 3-pass path narrows the two
    # products
    # before the cancelling subtract, so near-zero (cancellation) outputs lose relative
    # precision; the
    # ABSOLUTE error stays ~1 ULP of the product scale. Printed to stderr for the header perf table.
    xe = x[0::2].astype(np.float32)
    xo = x[1::2].astype(np.float32)
    ideal = np.empty(n, np.float32)
    ideal[0::2] = xe * c - xo * s
    ideal[1::2] = xe * s + xo * c
    ideal16 = ideal.astype(np.float16)
    abserr = np.abs(golden.astype(np.float64) - ideal16.astype(np.float64))
    sys.stderr.write(f"[rope datagen] N={n}: max |3pass - idealFP32| = {abserr.max():.4g} (abs); "
                     f"this is the accuracy cost of the zero-HW 3-pass decomposition.\n")

    emit = ["#include <stdint.h>"]
    emit += [format_scalar_definition("uint32_t", "rope_n", n)]
    emit += [format_scalar_definition("uint32_t", "rope_beats", beats)]
    emit += [
        format_vector_definition(
            "uint16_t", "rope_x", x.view(np.uint16),
            alignment=64, hex_bits=16, cast_hex=True,
        )
    ]
    emit += [
        format_vector_definition(
            "uint16_t", "rope_cos", cos_full.view(np.uint16),
            alignment=64, hex_bits=16, cast_hex=True,
        )
    ]
    emit += [
        format_vector_definition(
            "uint16_t", "rope_sin", sin_signed.view(np.uint16),
            alignment=64, hex_bits=16, cast_hex=True,
        )
    ]
    emit += [
        format_vector_definition(
            "uint16_t", "rope_golden", golden.view(np.uint16),
            alignment=64, hex_bits=16, cast_hex=True,
        )
    ]
    return "\n\n".join(emit)


def main():
    parser = argparse.ArgumentParser(description="Generating data for the SIMD RoPE kernel")
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
