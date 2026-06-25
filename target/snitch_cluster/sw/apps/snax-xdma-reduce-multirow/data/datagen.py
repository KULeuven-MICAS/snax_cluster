#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data generator for the standalone multi-row xDMA StreamReduce test. Emits ROWS independent FP16 rows
# of length D and the per-row FP16 goldens for SUMSQ / MAX / ADD -- the three reductions the llama3
# per-row RMSNorm/Softmax need. Each row gets its OWN data so a broken row-boundary re-init (a stale
# carry across rows) would show up as a wrong per-row scalar. The default rows use small values so the
# per-row SUMSQ stays inside the FP16 range of the narrowed reduce scalar.
#
# It ALSO emits a second, large-magnitude input (reduce_input_big, ~+-4096) whose per-row SUMSQ ~1e8-1e9
# is unrepresentable in fp16 (max finite 65504): the default fp16-out reduce returns garbage/inf, while
# the fp32out reduce (op CSR bit[9]) returns the true FP32 SUMSQ (reduce_ssq_big_f32_golden). The app
# runs both on this input to demonstrate the overflow fix.

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402

FP16_PER_BEAT = 32
RNG_SEED = 0x5EED


def emit_header_file(**kwargs):
    rows = int(kwargs["ROWS"])
    d = int(kwargs["D"])
    if d <= 0 or d % FP16_PER_BEAT != 0:
        raise ValueError(f"D must be a positive multiple of {FP16_PER_BEAT}, got {d}")
    if rows <= 0:
        raise ValueError(f"ROWS must be positive, got {rows}")
    beats = d // FP16_PER_BEAT

    rng = np.random.default_rng(RNG_SEED)
    xf = rng.uniform(-4.0, 4.0, size=(rows, d)).astype(np.float32)
    x16 = xf.astype(np.float16)            # snap onto the FP16 grid
    xf32 = x16.astype(np.float32)

    # Per-row reductions: FP32 accumulate, then narrow to FP16 (matches the HW reduce scalar).
    ssq16 = np.array([np.float16((xf32[r] ** 2).sum(dtype=np.float32)) for r in range(rows)])
    max16 = np.array([np.float16(xf32[r].max()) for r in range(rows)])
    sum16 = np.array([np.float16(xf32[r].sum(dtype=np.float32)) for r in range(rows)])

    x_u16 = x16.reshape(-1).view(np.uint16)

    # Large-magnitude input: per-row SUMSQ ~1e9 overflows the FP16 grid (max 65504) to +inf.
    xbig = rng.uniform(-4096.0, 4096.0, size=(rows, d)).astype(np.float32)
    xbig16 = xbig.astype(np.float16)            # snap onto the FP16 grid (4096 is representable)
    xbig32 = xbig16.astype(np.float32)
    ssq_big_f32 = np.array([(xbig32[r] ** 2).sum(dtype=np.float32) for r in range(rows)],
                           dtype=np.float32)
    # fp16 max finite is 65504, so this per-row SUMSQ is unrepresentable in fp16 (the bug); the HW's
    # fp16-out narrow yields garbage/inf, while the fp32out path delivers ssq_big_f32 below.
    xbig_u16 = xbig16.reshape(-1).view(np.uint16)

    emit = ["#include <stdint.h>"]
    emit += [format_scalar_definition("uint32_t", "reduce_rows", rows)]
    emit += [format_scalar_definition("uint32_t", "reduce_d", d)]
    emit += [format_scalar_definition("uint32_t", "reduce_beats", beats)]
    emit += [
        format_vector_definition("uint16_t", "reduce_input", x_u16,
                                 alignment=64, hex_bits=16, cast_hex=True)
    ]
    emit += [
        format_vector_definition("uint16_t", "reduce_ssq_golden", ssq16.view(np.uint16),
                                 alignment=64, hex_bits=16, cast_hex=True)
    ]
    emit += [
        format_vector_definition("uint16_t", "reduce_max_golden", max16.view(np.uint16),
                                 alignment=64, hex_bits=16, cast_hex=True)
    ]
    emit += [
        format_vector_definition("uint16_t", "reduce_sum_golden", sum16.view(np.uint16),
                                 alignment=64, hex_bits=16, cast_hex=True)
    ]
    # Large-input overflow fix: FP16-out golden (+inf) vs FP32-out golden (true SUMSQ bit pattern).
    emit += [
        format_vector_definition("uint16_t", "reduce_input_big", xbig_u16,
                                 alignment=64, hex_bits=16, cast_hex=True)
    ]
    emit += [
        format_vector_definition("uint32_t", "reduce_ssq_big_f32_golden", ssq_big_f32.view(np.uint32),
                                 alignment=64, hex_bits=32, cast_hex=True)
    ]
    return "\n\n".join(emit)


def main():
    parser = argparse.ArgumentParser(description="Generating data for the xDMA multi-row reduce test")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
