#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data generator for the local-loopback StreamMomentMergeRt smoke test (F3 S0a).
# Emits two pre-packed 512-bit (64B) input beats -- one FULL (nValid=8, no
# masking) and one SHORT (nValid=3, exercises the identity-lane masking) --
# plus the golden (m*, l*) for each, computed with the SAME online-softmax
# moment-merge monoid the RTL implements:
#   m* = max_i(m_i) ,  l* = sum_i( l_i * exp(m_i - m*) )
# (this is the standard flash-attention rescale fold; associative/commutative,
# so pairwise merge order doesn't matter -- see FpHelpers.scala:momentMerge).
#
# Beat layout (StreamMomentMergeRt.scala): 16 FP32 lanes; m_k = lane k,
# l_k = lane (maxPairs + k), maxPairs = 8. Lanes >= nValid are don't-care on
# input (the RTL substitutes the monoid identity), so we zero them here.

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa: E402

MAX_PAIRS = 8


def moment_merge_golden(m, l):
    """m, l: FP64 arrays of the SAME length (live lanes only). Returns (m*, l*) as FP32."""
    m = np.asarray(m, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    m_star = np.max(m)
    l_star = np.sum(l * np.exp(m - m_star))
    return np.float32(m_star), np.float32(l_star)


def pack_beat(m, l):
    """Pack up to MAX_PAIRS (m_k, l_k) FP32 pairs into one 16xFP32 (64B) beat."""
    beat = np.zeros(2 * MAX_PAIRS, dtype=np.float32)
    n = len(m)
    beat[0:n] = m
    beat[MAX_PAIRS:MAX_PAIRS + n] = l
    return beat.view(np.uint32)


def emit_header_file(**kwargs):
    seed = int(kwargs["seed"])
    rng = np.random.default_rng(seed)

    # ---- full beat: nValid=8, every lane live, no masking ----
    m8 = rng.uniform(-8.0, 8.0, size=8).astype(np.float32)
    l8 = rng.uniform(0.01, 4.0, size=8).astype(np.float32)
    m8_star, l8_star = moment_merge_golden(m8, l8)
    beat8 = pack_beat(m8, l8)

    # ---- short beat: nValid=3, lanes 3..7 must be masked to the identity ----
    m3 = rng.uniform(-8.0, 8.0, size=3).astype(np.float32)
    l3 = rng.uniform(0.01, 4.0, size=3).astype(np.float32)
    m3_star, l3_star = moment_merge_golden(m3, l3)
    beat3 = pack_beat(m3, l3)

    emit = ["#include <stdint.h>"]
    emit += [format_vector_definition("uint32_t", "mmerge_beat8_in", beat8,
                                      alignment=64, hex_bits=32, cast_hex=True)]
    emit += [format_scalar_definition("uint32_t", "mmerge_beat8_nvalid", 8)]
    emit += [format_scalar_definition("uint32_t", "mmerge_beat8_m_golden",
                                      int(m8_star.view(np.uint32)))]
    emit += [format_scalar_definition("uint32_t", "mmerge_beat8_l_golden",
                                      int(l8_star.view(np.uint32)))]

    emit += [format_vector_definition("uint32_t", "mmerge_beat3_in", beat3,
                                      alignment=64, hex_bits=32, cast_hex=True)]
    emit += [format_scalar_definition("uint32_t", "mmerge_beat3_nvalid", 3)]
    emit += [format_scalar_definition("uint32_t", "mmerge_beat3_m_golden",
                                      int(m3_star.view(np.uint32)))]
    emit += [format_scalar_definition("uint32_t", "mmerge_beat3_l_golden",
                                      int(l3_star.view(np.uint32)))]
    return "\n\n".join(emit)


def main():
    parser = argparse.ArgumentParser(
        description="Data for the local-loopback StreamMomentMergeRt smoke test")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
