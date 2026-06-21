#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data generator for the xDMA FP16 softmax test. Emits one FP16 row of length N
# (a multiple of 32, no padding) and the FP16 golden softmax.

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
    xf = rng.uniform(-4.0, 4.0, size=n).astype(np.float32)
    x16 = xf.astype(np.float16)            # snap inputs onto the FP16 grid
    xf32 = x16.astype(np.float32)

    m = np.float32(xf32.max())
    e = np.exp((xf32 - m).astype(np.float64))
    s = e.sum()
    out16 = (e / s).astype(np.float16)     # golden softmax, narrowed to FP16

    # golden exp(x) for the standalone StreamMap+FpExp 1:1 benchmark (no reduction)
    exp16 = np.exp(xf32.astype(np.float64)).astype(np.float16)

    x_u16 = x16.view(np.uint16)
    g_u16 = out16.view(np.uint16)
    e_u16 = exp16.view(np.uint16)

    emit = ["#include <stdint.h>"]
    emit += [format_scalar_definition("uint32_t", "softmax_n", n)]
    emit += [format_scalar_definition("uint32_t", "softmax_beats", beats)]
    emit += [
        format_vector_definition(
            "uint16_t", "softmax_input", x_u16,
            alignment=64, hex_bits=16, cast_hex=True,
        )
    ]
    emit += [
        format_vector_definition(
            "uint16_t", "softmax_golden", g_u16,
            alignment=64, hex_bits=16, cast_hex=True,
        )
    ]
    emit += [
        format_vector_definition(
            "uint16_t", "exp_golden", e_u16,
            alignment=64, hex_bits=16, cast_hex=True,
        )
    ]
    return "\n\n".join(emit)


def main():
    parser = argparse.ArgumentParser(description="Generating data for the xDMA softmax kernel")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
