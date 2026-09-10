#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data generator for the multi-row xDMA FP16 softmax A/B test: per row,
# out[r,:] = exp(x[r,:] - max[r]) / Sexp[r]. Same tensor + golden as the
# snax-simd-softmax-multirow kernel (identical RNG seed, and each stage narrowed to FP16
# exactly as the HW does), so the two apps are directly comparable; adds the per-tensor
# int8 quant scale used by the fused StreamElementwise -||> Fp16ToInt8 pass.
# The DM core has no FPU, so the per-row reciprocal 1/Sexp is precomputed here.

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
    if (rows * beats) % 2 != 0:
        raise ValueError("rows*beats must be even (Fp16ToInt8 packs 2 FP16 beats -> 1 INT8 beat)")

    rng = np.random.default_rng(RNG_SEED)
    xf = rng.uniform(-4.0, 4.0, size=(rows, d)).astype(np.float32)
    x16 = xf.astype(np.float16)            # snap onto the FP16 grid
    xf32 = x16.astype(np.float32)

    inv_sum16 = np.empty(rows, dtype=np.float16)
    out16 = np.empty((rows, d), dtype=np.float16)
    for r in range(rows):
        max16 = np.float16(xf32[r].max())
        xs16 = (xf32[r] - max16.astype(np.float32)).astype(np.float16)   # sew(ADD) FP16 out
        expb16 = np.exp(xs16.astype(np.float64)).astype(np.float16)      # smap(EXP) FP16 out
        sum16 = np.float16(expb16.astype(np.float32).sum(dtype=np.float32))  # reduce(ADD)
        inv_sum16[r] = np.float16(np.float32(1.0) / sum16.astype(np.float32))
        out16[r] = (expb16.astype(np.float32) * inv_sum16[r].astype(np.float32)).astype(np.float16)

    # Per-tensor symmetric int8 scale for the fused quant (zero-point 0, range [-127,127]).
    amax = np.float32(np.abs(out16.astype(np.float32)).max())
    inv_scale = np.float32(127.0) / amax
    inv_scale_bits = int(inv_scale.view(np.uint32))

    emit = ["#include <stdint.h>"]
    emit += [format_scalar_definition("uint32_t", "smf_rows", rows)]
    emit += [format_scalar_definition("uint32_t", "smf_d", d)]
    emit += [format_scalar_definition("uint32_t", "smf_beats", beats)]
    emit += [format_scalar_definition("uint32_t", "smf_inv_scale", inv_scale_bits)]
    emit += [
        format_vector_definition("uint16_t", "smf_input", x16.reshape(-1).view(np.uint16),
                                 alignment=64, hex_bits=16, cast_hex=True)
    ]
    emit += [
        format_vector_definition("uint16_t", "smf_inv_sum", inv_sum16.view(np.uint16),
                                 alignment=64, hex_bits=16, cast_hex=True)
    ]
    emit += [
        format_vector_definition("uint16_t", "smf_out_golden", out16.reshape(-1).view(np.uint16),
                                 alignment=64, hex_bits=16, cast_hex=True)
    ]
    return "\n\n".join(emit)


def main():
    parser = argparse.ArgumentParser(
        description="Generating data for the multi-row xDMA softmax fold (A/B) kernel")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
