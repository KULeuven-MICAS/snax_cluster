#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data generator for the RUNTIME-PRECISION xDMA SiLU test. Emits ONE row of N
# values, snapped onto BOTH the FP16 grid and the FP8 (e5m2) grid, plus the
# matching gelu goldens. The SAME xDMA netlist (StreamMapRt with func=GELU) runs
# the whole (unary, single-pass) kernel at either precision, chosen at runtime by
# the `fmt` CSR field -- so the host test feeds the FP16 dataset with fmt=FP16 and
# the FP8 dataset with fmt=FP8 and checks each against its own grid-faithful
# golden.
#
# The goldens MIRROR the hardware datapath exactly (FP32-internal, values snapped
# at the stage boundary): the grid-snapped input is fed to one StreamMapRt GELU
# pass -- gelu(x) = x * sigmoid(x) = x / (1 + exp(-x)), computed in f64 -- and the
# result is narrowed back to the transport format. gelu is signed (small negative
# for x < 0), so the goldens carry the sign as well.

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402

FP16_PER_BEAT = 32
FP8_PER_BEAT = 64
RNG_SEED = 320


# ---- FP8 e5m2 codec (1 sign, 5 exp bias 15, 2 mant), RNE + subnormal flush-to-zero -------------
def f32_to_e5m2_bits(x):
    x = np.asarray(x, dtype=np.float32)
    a = np.abs(x).astype(np.float64)
    sign = np.where(np.signbit(x), 0x80, 0).astype(np.int32)
    nz = a > 0.0
    e = np.floor(np.log2(np.where(nz, a, 1.0))).astype(np.int32)
    exp = e + 15
    frac = np.where(nz, a / np.power(2.0, e.astype(np.float64)) - 1.0, 0.0)
    mant = np.rint(frac * 4.0).astype(np.int32)           # RNE
    carry = mant == 4
    mant = np.where(carry, 0, mant)
    exp = np.where(carry, exp + 1, exp)
    under = (~nz) | (exp < 1)
    over = exp >= 0x1F
    bits = sign | (exp << 2) | mant
    bits = np.where(over, sign | 0x7B, bits)              # clamp to max finite (57344)
    bits = np.where(under, 0, bits)                       # FTZ (matches narrowRt)
    return bits.astype(np.uint8)


def e5m2_bits_to_f32(b):
    b = np.asarray(b, dtype=np.uint8).astype(np.int32)
    sign = np.where((b & 0x80) != 0, -1.0, 1.0)
    exp = (b >> 2) & 0x1F
    mant = b & 0x3
    val = np.where(exp == 0,
                   sign * mant * (2.0 ** -16),
                   sign * (4 + mant) * np.power(2.0, (exp - 17).astype(np.float64)))
    return val.astype(np.float32)



def _erf(x):
    x = np.asarray(x, dtype=np.float64)
    t = 1.0 / (1.0 + 0.3275911 * np.abs(x))
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return np.where(x >= 0, y, -y)


def _phi(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=np.float64) / np.sqrt(2.0)))


def gelu_chain_fp16(xf):
    x16 = xf.astype(np.float16)                           # snap input onto the FP16 grid
    x32 = x16.astype(np.float32).astype(np.float64)
    sig = _phi(x32)
    y16 = (x32 * sig).astype(np.float16)                  # map GELU output (fmt-narrowed)
    return x16.view(np.uint16), y16.view(np.uint16)


def gelu_chain_fp8(xf):
    x8 = f32_to_e5m2_bits(xf)                             # snap input onto the FP8 (e5m2) grid
    x32 = e5m2_bits_to_f32(x8)
    xd = x32.astype(np.float64)
    sig = _phi(xd)
    gelu_f32 = (xd * sig).astype(np.float32)
    y8 = f32_to_e5m2_bits(gelu_f32)                       # map GELU output (fmt-narrowed)
    return x8, y8


def emit_header_file(**kwargs):
    n = int(kwargs["N"])
    if n <= 0 or n % FP8_PER_BEAT != 0:
        raise ValueError(f"N must be a positive multiple of {FP8_PER_BEAT}, got {n}")

    rng = np.random.default_rng(RNG_SEED)
    xf = rng.uniform(-4.0, 4.0, size=n).astype(np.float32)

    x16u, g16u = gelu_chain_fp16(xf)
    x8u, g8u = gelu_chain_fp8(xf)

    emit = ["#include <stdint.h>"]
    emit += [format_scalar_definition("uint32_t", "gelu_n", n)]
    emit += [format_scalar_definition("uint32_t", "gelu_beats_fp16", n // FP16_PER_BEAT)]
    emit += [format_scalar_definition("uint32_t", "gelu_beats_fp8", n // FP8_PER_BEAT)]
    # FP16 dataset
    emit += [format_vector_definition("uint16_t", "gelu_input_fp16", x16u,
                                      alignment=64, hex_bits=16, cast_hex=True)]
    emit += [format_vector_definition("uint16_t", "gelu_golden_fp16", g16u,
                                      alignment=64, hex_bits=16, cast_hex=True)]
    # FP8 dataset
    emit += [format_vector_definition("uint8_t", "gelu_input_fp8", x8u,
                                      alignment=64, hex_bits=8, cast_hex=True)]
    emit += [format_vector_definition("uint8_t", "gelu_golden_fp8", g8u,
                                      alignment=64, hex_bits=8, cast_hex=True)]
    return "\n\n".join(emit)


def main():
    parser = argparse.ArgumentParser(description="Data for the runtime-precision xDMA gelu kernel")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
