#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data generator for the runtime-precision xDMA MoE-gating (argmax) test. One beat
# of E expert logits (E=32 FP16 or E=64 FP8, both = 64 bytes) + the golden argmax
# expert index, at both precisions. The argmax is computed on the grid-snapped
# logits so it matches what the hardware sees.

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402

RNG_SEED = 411


def f32_to_e5m2_bits(x):
    x = np.asarray(x, dtype=np.float32)
    a = np.abs(x).astype(np.float64)
    sign = np.where(np.signbit(x), 0x80, 0).astype(np.int32)
    nz = a > 0.0
    e = np.floor(np.log2(np.where(nz, a, 1.0))).astype(np.int32)
    exp = e + 15
    frac = np.where(nz, a / np.power(2.0, e.astype(np.float64)) - 1.0, 0.0)
    mant = np.rint(frac * 4.0).astype(np.int32)
    carry = mant == 4
    mant = np.where(carry, 0, mant)
    exp = np.where(carry, exp + 1, exp)
    under = (~nz) | (exp < 1)
    over = exp >= 0x1F
    bits = sign | (exp << 2) | mant
    bits = np.where(over, sign | 0x7B, bits)
    bits = np.where(under, 0, bits)
    return bits.astype(np.uint8)


def e5m2_bits_to_f32(b):
    b = np.asarray(b, dtype=np.uint8).astype(np.int32)
    sign = np.where((b & 0x80) != 0, -1.0, 1.0)
    exp = (b >> 2) & 0x1F
    mant = b & 0x3
    val = np.where(exp == 0, sign * mant * (2.0 ** -16),
                   sign * (4 + mant) * np.power(2.0, (exp - 17).astype(np.float64)))
    return val.astype(np.float32)


def emit_header_file(**kwargs):
    _ = kwargs
    rng = np.random.default_rng(RNG_SEED)
    # FP16: 32 experts (64-byte beat)
    lf16 = rng.uniform(-4.0, 4.0, size=32).astype(np.float16)
    arg16 = int(np.argmax(lf16.astype(np.float32)))
    # FP8: 64 experts (64-byte beat), on the e5m2 grid
    l8bits = f32_to_e5m2_bits(rng.uniform(-4.0, 4.0, size=64).astype(np.float32))
    arg8 = int(np.argmax(e5m2_bits_to_f32(l8bits)))

    emit = ["#include <stdint.h>"]
    emit += [format_scalar_definition("uint32_t", "moe_e_fp16", 32)]
    emit += [format_scalar_definition("uint32_t", "moe_e_fp8", 64)]
    emit += [format_scalar_definition("uint32_t", "moe_argmax_fp16", arg16)]
    emit += [format_scalar_definition("uint32_t", "moe_argmax_fp8", arg8)]
    emit += [format_vector_definition("uint16_t", "moe_logits_fp16", lf16.view(np.uint16),
                                      alignment=64, hex_bits=16, cast_hex=True)]
    emit += [format_vector_definition("uint8_t", "moe_logits_fp8", l8bits,
                                      alignment=64, hex_bits=8, cast_hex=True)]
    return "\n\n".join(emit)


def main():
    parser = argparse.ArgumentParser(description="Data for the runtime-precision xDMA MoE-gating kernel")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
