#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data generator for the xDMA FP16 SiLU test. Emits one FP16 row of length N (a multiple of 32) and the
# FP16 golden silu(x) = x * sigmoid(x) = x / (1 + e^-x). Inputs are snapped onto the FP16 grid first so the
# golden reflects exactly what the hardware consumes.

import argparse
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
    beats = n // FP16_PER_BEAT

    rng = np.random.default_rng(RNG_SEED)
    xf = rng.uniform(-8.0, 8.0, size=n).astype(np.float32)
    x16 = xf.astype(np.float16)            # snap inputs onto the FP16 grid
    xf32 = x16.astype(np.float32).astype(np.float64)

    sig = 1.0 / (1.0 + np.exp(-xf32))
    out16 = (xf32 * sig).astype(np.float16)  # golden silu -> FP16

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
    return "\n\n".join(emit)


def main():
    parser = argparse.ArgumentParser(description="Generating data for the xDMA silu kernel")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
