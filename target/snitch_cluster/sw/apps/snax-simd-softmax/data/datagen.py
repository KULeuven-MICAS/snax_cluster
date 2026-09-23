#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data for the softmax kernel: one [rows, cols] FP16 tile, emitted BOTH ways round.
#
# x    [rows, cols]  row-major -- one token per row, the layout the layer hands the kernel
# x_t  [cols, rows]  transposed -- one token per FP16 LANE, the layout that makes BOTH
#                    reductions (max and sum) run along beats instead of across lanes
#
# Same numbers, same goldens, so the paths in the app are directly comparable.
#
# TWO GOLDENS, because the chain narrows to FP16 at every stage and the exact softmax
# does not:
#
#   out_golden  the FP16 chain: fp16(x-max) -> fp16(exp) -> fp16(sum) -> fp16(1/sum) ->
#               fp16(exp * inv). What the device computes apart from its exp LUT, and the
#               reference the deployed path is scored against.
#   out_exact   softmax in FP64, narrowed once at the end. The truth.
#
# inv_sum is the HOST reciprocal the deployed kernel needs precomputed, because an
# rv32ima core has no FPU and the SIMD block had no reciprocal. The app computes its own
# on the device instead (rsqrt of the square); this stays so the legacy path it replaces
# can still be run and timed.

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
FP16_MAX = 65504.0


def fp16_mono(h: int) -> int:
    mag = h & 0x7FFF
    return (0x8000 - mag) if (h & 0x8000) else (0x8000 + mag)


def emit_header_file(**kwargs):
    rows = int(kwargs["ROWS"])
    d = int(kwargs["D"])
    if d <= 0 or d % FP16_PER_BEAT != 0:
        raise ValueError(f"D must be a positive multiple of {FP16_PER_BEAT}, got {d}")
    if rows != FP16_PER_BEAT:
        raise ValueError(
            f"ROWS must be {FP16_PER_BEAT}: the transposed path puts one token per FP16 "
            f"lane, so a whole tile's per-token scalars are exactly one beat. Got {rows}.")
    beats = d // FP16_PER_BEAT

    rng = np.random.default_rng(RNG_SEED)
    x16 = rng.uniform(-4.0, 4.0, size=(rows, d)).astype(np.float32).astype(np.float16)
    xf32 = x16.astype(np.float32)

    # ---- the FP16 chain, stage by stage, exactly as the device runs it ----
    mx16 = x16.max(axis=1)                                    # max of FP16 values: exact
    xs16 = (xf32 - mx16.astype(np.float32)[:, None]).astype(np.float16)
    ex16 = np.exp(xs16.astype(np.float64)).astype(np.float16)  # device uses a LUT here
    # The reduce accumulates in FP32 and narrows the row scalar to FP16 -- so does this.
    s16 = np.array([np.float16(ex16[r].astype(np.float32).sum(dtype=np.float32))
                    for r in range(rows)], dtype=np.float16)
    inv16 = (np.float32(1.0) / s16.astype(np.float32)).astype(np.float16)

    out_chain = np.empty((rows, d), dtype=np.float16)
    out_exact = np.empty((rows, d), dtype=np.float16)
    for r in range(rows):
        out_chain[r] = (ex16[r].astype(np.float32) * inv16[r].astype(np.float32)).astype(np.float16)
        e = np.exp((xf32[r] - np.float64(mx16[r])).astype(np.float64))
        out_exact[r] = (e / e.sum()).astype(np.float16)

    # THE RANGE LIMIT ON THE ON-DEVICE RECIPROCAL. 1/s is computed as rsqrt(s*s), and the
    # transport between two chained operators is FP16, so s*s must land inside FP16. Each
    # term of s is exp(x-max) <= 1, so s <= D and the bound is D^2 <= 65504, i.e. D <= 255
    # for a pathological all-equal row. Report what this tile actually reaches so the app
    # can print the headroom rather than assume it.
    sq_max = float(np.max(s16.astype(np.float32) ** 2))
    if sq_max > FP16_MAX:
        raise ValueError(
            f"max(sum^2) = {sq_max:.0f} overflows FP16: the rsqrt(s*s) reciprocal cannot "
            f"be used at this shape. Use the rsqrt-then-square order instead (see the "
            f"note in the app header) or reduce D.")

    # INT8 quantise golden for the Fp16ToInt8 stage: q = sat127(rne(fp32(out)*inv_scale)).
    # Softmax probabilities live in [0,1], so inv_scale=127 maps them onto [0,127].
    # Mirrors the HW PE (fp32 product, clamp to +/-128 before the RNE round, symmetric sat).
    inv_scale = np.float32(127.0)
    prod = np.clip(out_exact.astype(np.float32) * inv_scale, np.float32(-128.0), np.float32(128.0))
    q_i8 = np.clip(np.rint(prod.astype(np.float64)), -127, 127).astype(np.int8)

    # How far the FP16 chain sits from the exact softmax, before the device's exp LUT adds
    # anything -- reported here so the app does not have to rediscover the floor.
    worst_chain_ulp = 0
    for r in range(rows):
        for c in range(d):
            g = out_exact[r, c].view(np.uint16)
            if (int(g) & 0x7C00) == 0:
                continue
            u = abs(fp16_mono(int(out_chain[r, c].view(np.uint16))) - fp16_mono(int(g)))
            worst_chain_ulp = max(worst_chain_ulp, u)

    emit = ["#include <stdint.h>",
            f"// FP16 chain vs exact softmax: worst {worst_chain_ulp} FP16 ULP (the device's "
            f"exp LUT adds to this)",
            f"// max(sum^2) = {sq_max:.0f} of FP16's {FP16_MAX:.0f} -- the headroom the "
            f"rsqrt(s*s) reciprocal runs on"]
    emit += [format_scalar_definition("uint32_t", "sm_rows", rows)]
    emit += [format_scalar_definition("uint32_t", "sm_d", d)]
    emit += [format_scalar_definition("uint32_t", "sm_beats", beats)]
    emit += [format_scalar_definition("uint32_t", "sm_chain_ulp", worst_chain_ulp)]
    emit += [format_scalar_definition("uint32_t", "sm_inv_scale", int(inv_scale.view(np.uint32)))]

    def vec(name, arr):
        return format_vector_definition("uint16_t", name, arr, alignment=64,
                                        hex_bits=16, cast_hex=True)

    emit += [vec("sm_input", x16.reshape(-1).view(np.uint16))]
    emit += [vec("sm_input_t", x16.T.copy().reshape(-1).view(np.uint16))]
    emit += [vec("sm_max_golden", mx16.view(np.uint16))]
    emit += [vec("sm_sum_golden", s16.view(np.uint16))]
    emit += [vec("sm_inv_sum", inv16.view(np.uint16))]
    emit += [vec("sm_out_golden", out_chain.reshape(-1).view(np.uint16))]
    emit += [vec("sm_out_exact", out_exact.reshape(-1).view(np.uint16))]
    emit += [format_vector_definition("int8_t", "sm_golden_i8", q_i8.reshape(-1), alignment=64)]
    return "\n\n".join(emit)


def main():
    parser = argparse.ArgumentParser(description="Data for the SIMD FP16 softmax kernel")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
