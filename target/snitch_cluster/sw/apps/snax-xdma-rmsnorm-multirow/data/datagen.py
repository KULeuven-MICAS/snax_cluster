#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data generator for the multi-row xDMA FP16 RMSNorm test: per row, out[r,:] = x[r,:] * inv_rms[r],
# inv_rms[r] = 1/sqrt(mean(x[r,:]^2)). This is the per-row (multi-row) form the llama3 layer needs --
# the single-row app applies one StreamMap scalar to the whole row; here each row has its OWN inv_rms,
# so the runtime does: multi-row StreamReduce(SUMSQ) -> DM-core scalar broadcast inv_rms[r]->[r,:] ->
# StreamElementwise(MUL). The DM core has no FPU, so inv_rms (an rsqrt) is precomputed here and matches
# the HW by being derived from the FP16-narrowed per-row Sx^2 (mean = Sx^2/D is exact for D=2^k).

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
RNG_SEED = 0x12345


def emit_header_file(**kwargs):
    rows = int(kwargs["ROWS"])
    d = int(kwargs["D"])
    if d <= 0 or d % FP16_PER_BEAT != 0:
        raise ValueError(f"D must be a positive multiple of {FP16_PER_BEAT}, got {d}")
    if d & (d - 1) != 0:
        raise ValueError(f"D must be a power of two (for the /D exponent trick), got {d}")
    if rows <= 0:
        raise ValueError(f"ROWS must be positive, got {rows}")
    beats = d // FP16_PER_BEAT

    rng = np.random.default_rng(RNG_SEED)
    xf = rng.uniform(-4.0, 4.0, size=(rows, d)).astype(np.float32)
    x16 = xf.astype(np.float16)            # snap onto the FP16 grid
    xf32 = x16.astype(np.float32)

    inv_rms16 = np.empty(rows, dtype=np.float16)
    out16 = np.empty((rows, d), dtype=np.float16)
    for r in range(rows):
        ssq16 = np.float16((xf32[r] ** 2).sum(dtype=np.float32))  # FP32 accumulate -> FP16 (HW reduce scalar)
        mean = ssq16.astype(np.float32) / np.float32(d)           # exact: D = 2^k
        inv_rms16[r] = np.float16(np.float32(1.0) / np.float32(np.sqrt(mean)))
        out16[r] = (xf32[r] * inv_rms16[r].astype(np.float32)).astype(np.float16)

    emit = ["#include <stdint.h>"]
    emit += [format_scalar_definition("uint32_t", "rmsmr_rows", rows)]
    emit += [format_scalar_definition("uint32_t", "rmsmr_d", d)]
    emit += [format_scalar_definition("uint32_t", "rmsmr_beats", beats)]
    emit += [
        format_vector_definition("uint16_t", "rmsmr_input", x16.reshape(-1).view(np.uint16),
                                 alignment=64, hex_bits=16, cast_hex=True)
    ]
    emit += [
        format_vector_definition("uint16_t", "rmsmr_inv_rms", inv_rms16.view(np.uint16),
                                 alignment=64, hex_bits=16, cast_hex=True)
    ]
    emit += [
        format_vector_definition("uint16_t", "rmsmr_out_golden", out16.reshape(-1).view(np.uint16),
                                 alignment=64, hex_bits=16, cast_hex=True)
    ]
    return "\n\n".join(emit)


def main():
    parser = argparse.ArgumentParser(description="Generating data for the multi-row xDMA rmsnorm kernel")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
