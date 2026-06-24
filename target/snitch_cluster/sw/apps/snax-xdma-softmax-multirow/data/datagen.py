#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data generator for the multi-row xDMA FP16 softmax test: per row, out[r,:] = softmax(x[r,:]). The
# golden MIRRORS the multi-row HW chain step by step so the FP16-ULP compare stays tight -- the multi-row
# path narrows to FP16 at every stage (sew x-max, smap exp, reduce sum, sew *recip), one more narrowing
# than the single-row fused StreamMap path:
#   max16 -> xs16 = fp16(x - max) -> expb16 = fp16(exp(xs16)) -> sum16 -> inv_sum16 = fp16(1/sum16) ->
#   out16 = fp16(expb16 * inv_sum16).
# -max is an integer FP16 sign flip at runtime (no FPU); the reciprocal is precomputed here (on HeMAiA
# the host CVA6 does it via host_scalar_bcast).

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402

FP16_PER_BEAT = 32
RNG_SEED = 0x2BCD


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

    inv_sum16 = np.empty(rows, dtype=np.float16)
    out16 = np.empty((rows, d), dtype=np.float16)
    for r in range(rows):
        max16 = np.float16(xf32[r].max())
        xs16 = (xf32[r] - max16.astype(np.float32)).astype(np.float16)        # sew(ADD) FP16 output
        expb16 = np.exp(xs16.astype(np.float64)).astype(np.float16)           # smap(EXP) FP16 output
        sum16 = np.float16(expb16.astype(np.float32).sum(dtype=np.float32))   # reduce(ADD) FP16 scalar
        inv_sum16[r] = np.float16(np.float32(1.0) / sum16.astype(np.float32))  # FP16 reciprocal (bcast)
        out16[r] = (expb16.astype(np.float32) * inv_sum16[r].astype(np.float32)).astype(np.float16)

    emit = ["#include <stdint.h>"]
    emit += [format_scalar_definition("uint32_t", "smr_rows", rows)]
    emit += [format_scalar_definition("uint32_t", "smr_d", d)]
    emit += [format_scalar_definition("uint32_t", "smr_beats", beats)]
    emit += [
        format_vector_definition("uint16_t", "smr_input", x16.reshape(-1).view(np.uint16),
                                 alignment=64, hex_bits=16, cast_hex=True)
    ]
    emit += [
        format_vector_definition("uint16_t", "smr_inv_sum", inv_sum16.view(np.uint16),
                                 alignment=64, hex_bits=16, cast_hex=True)
    ]
    emit += [
        format_vector_definition("uint16_t", "smr_out_golden", out16.reshape(-1).view(np.uint16),
                                 alignment=64, hex_bits=16, cast_hex=True)
    ]
    return "\n\n".join(emit)


def main():
    parser = argparse.ArgumentParser(description="Generating data for the multi-row xDMA softmax kernel")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
