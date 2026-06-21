#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data generator for the xDMA FP16 SwiGLU test. Emits two FP16 rows of length N (a multiple of 32):
# `gate` and `up`. The xDMA offloads sg = silu(gate); the host then forms out = sg * up. We emit both the
# silu(gate) golden (verified on-cluster) and the full swiglu golden out = silu(gate)*up (the host's
# reference). Inputs are snapped onto the FP16 grid first so the goldens reflect what the hardware consumes.

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
    gate = rng.uniform(-8.0, 8.0, size=n).astype(np.float16)
    up = rng.uniform(-2.0, 2.0, size=n).astype(np.float16)
    gate32 = gate.astype(np.float64)
    up32 = up.astype(np.float64)

    sg = (gate32 / (1.0 + np.exp(-gate32)))         # silu(gate), FP64
    sg16 = sg.astype(np.float16)                     # on-cluster xDMA golden
    out16 = (sg16.astype(np.float64) * up32).astype(np.float16)  # host (.)up: out = silu(gate)*up

    emit = ["#include <stdint.h>"]
    emit += [format_scalar_definition("uint32_t", "swiglu_n", n)]
    emit += [format_scalar_definition("uint32_t", "swiglu_beats", beats)]
    emit += [
        format_vector_definition(
            "uint16_t", "swiglu_gate", gate.view(np.uint16),
            alignment=64, hex_bits=16, cast_hex=True,
        )
    ]
    emit += [
        format_vector_definition(
            "uint16_t", "swiglu_up", up.view(np.uint16),
            alignment=64, hex_bits=16, cast_hex=True,
        )
    ]
    emit += [
        format_vector_definition(
            "uint16_t", "swiglu_silu_gate_golden", sg16.view(np.uint16),
            alignment=64, hex_bits=16, cast_hex=True,
        )
    ]
    emit += [
        format_vector_definition(
            "uint16_t", "swiglu_golden", out16.view(np.uint16),
            alignment=64, hex_bits=16, cast_hex=True,
        )
    ]
    return "\n\n".join(emit)


def main():
    parser = argparse.ArgumentParser(description="Generating data for the xDMA swiglu kernel")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
