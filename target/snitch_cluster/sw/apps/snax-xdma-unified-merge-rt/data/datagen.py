#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data generator for the local-loopback UnifiedMonoidMergeRt smoke test. ONE
# configurable combine cell is exercised in every mode of the arithmetic family,
# each computed with the SAME monoid the RTL implements:
#   MOMENT  (softmax normalizer)  m* = max_i m_i ,  l* = Σ_i l_i·exp(m_i - m*)
#   SUM     (LayerNorm/RMSNorm)   (S1*, S2*) = (Σ_i S1_i, Σ_i S2_i)
#   MAXPOOL (max-reduce)          m* = max_i m_i          (field1 lanes ignored)
#   ATTN    (m, ℓ, O) triple      m* = max ; ℓ*,O* rescaled by exp(m_i - m*)
#
# Paired-mode beat layout (field0_k = lane k, field1_k = lane 8+k, 16 FP32 lanes):
#   MOMENT/SUM/MAXPOOL pack 8 (field0, field1) partials into one 64B beat.
# ATTN beat layout: lane0 = m, lane1 = ℓ, lanes 2..2+dHead-1 = O ; folded across
# TWO beats via the accEn slot (arm the slot on beat 0, fold beat 1 in).

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa: E402

MAX_PAIRS = 8
DHEAD = 8


def pack_pairs(f0, f1):
    """Pack up to MAX_PAIRS (field0, field1) FP32 partials into one 16xFP32 (64B) beat."""
    beat = np.zeros(2 * MAX_PAIRS, dtype=np.float32)
    n = len(f0)
    beat[0:n] = f0
    beat[MAX_PAIRS:MAX_PAIRS + n] = f1
    return beat.view(np.uint32)


def pack_attn(m, l, o):
    """Pack one (m, ℓ, O[DHEAD]) partial: lane0=m, lane1=ℓ, lanes 2.. = O."""
    beat = np.zeros(2 * MAX_PAIRS, dtype=np.float32)
    beat[0] = m
    beat[1] = l
    beat[2:2 + len(o)] = o  # payload length varies: dHead for ATTN, 2 (A,B) for MOMENT2
    return beat.view(np.uint32)


def u32(x):
    return int(np.float32(x).view(np.uint32))


def emit_header_file(**kwargs):
    rng = np.random.default_rng(int(kwargs["seed"]))
    emit = ["#include <stdint.h>", format_scalar_definition("uint32_t", "unified_dhead", DHEAD)]

    # ---- MOMENT: 8 local (m_k, l_k) shard stats -> global (m*, l*) ----
    m8 = rng.uniform(-8.0, 8.0, size=8).astype(np.float32).astype(np.float64)
    l8 = rng.uniform(0.5, 4.0, size=8).astype(np.float32).astype(np.float64)
    mm = np.max(m8)
    ml = np.sum(l8 * np.exp(m8 - mm))
    emit += [format_vector_definition("uint32_t", "moment_beat_in", pack_pairs(
        m8.astype(np.float32), l8.astype(np.float32)), alignment=64, hex_bits=32, cast_hex=True)]
    emit += [format_scalar_definition("uint32_t", "moment_m_golden", u32(mm))]
    emit += [format_scalar_definition("uint32_t", "moment_l_golden", u32(ml))]

    # ---- SUM: 8 (S1_k, S2_k) partials -> (ΣS1, ΣS2) ----
    s1 = rng.uniform(-4.0, 4.0, size=8).astype(np.float32)
    s2 = rng.uniform(0.0, 8.0, size=8).astype(np.float32)
    emit += [format_vector_definition("uint32_t", "sum_beat_in", pack_pairs(s1, s2),
                                      alignment=64, hex_bits=32, cast_hex=True)]
    emit += [format_scalar_definition("uint32_t", "sum_s1_golden", u32(np.sum(s1.astype(np.float64))))]
    emit += [format_scalar_definition("uint32_t", "sum_s2_golden", u32(np.sum(s2.astype(np.float64))))]

    # ---- MAXPOOL: max over 8 field0 lanes (field1 don't-care) ----
    v8 = rng.uniform(-9.0, 9.0, size=8).astype(np.float32)
    j8 = rng.uniform(-1.0, 1.0, size=8).astype(np.float32)
    emit += [format_vector_definition("uint32_t", "maxpool_beat_in", pack_pairs(v8, j8),
                                      alignment=64, hex_bits=32, cast_hex=True)]
    emit += [format_scalar_definition("uint32_t", "maxpool_m_golden", u32(np.max(v8)))]

    # ---- ARGMAX: 8 (logit, idx) candidates -> max logit + its carried index (top-1) ----
    # distinct logits so the argmax is unambiguous (WSEL ties break to the first operand). Force the max to a
    # NONZERO index so the carried-index check is meaningful (a zero idx would alias a zeroed/masked field).
    perm = rng.permutation(8)
    logits = (-4.0 + perm.astype(np.float32) * 1.1).astype(np.float32)
    win = 5  # put the largest logit at index 5
    hi = int(np.argmax(logits))
    logits[hi], logits[win] = logits[win], logits[hi]
    idxf = np.arange(8, dtype=np.float32)  # field1 carries the index (as a float, passed through verbatim)
    gmax = float(np.max(logits))
    gidx = float(int(np.argmax(logits)))
    emit += [format_vector_definition("uint32_t", "argmax_beat_in", pack_pairs(logits, idxf),
                                      alignment=64, hex_bits=32, cast_hex=True)]
    emit += [format_scalar_definition("uint32_t", "argmax_m_golden", u32(gmax))]
    emit += [format_scalar_definition("uint32_t", "argmax_idx_golden", u32(gidx))]

    # ---- MOMENT2: two shard partials (m, ℓ=Σeˢ, A=Σeˢv, B=Σeˢv²) -> exp-weighted moment bank ----
    # B (the 2nd moment) is produced upstream by the SIMD egress square feature; here the beats carry the
    # pre-computed per-shard partials, exactly as MOMENT/ATTN carry pre-summed statistics.
    m2 = []
    for _ in range(2):
        m = float(rng.uniform(-2.0, 5.0))
        ll = float(rng.uniform(1.0, 5.0))
        a = float(rng.uniform(-4.0, 4.0))
        b = float(rng.uniform(0.5, 6.0))
        m2.append((m, ll, a, b))
    m2ms = max(s[0] for s in m2)
    m2l = sum(s[1] * np.exp(s[0] - m2ms) for s in m2)
    m2a = sum(s[2] * np.exp(s[0] - m2ms) for s in m2)
    m2b = sum(s[3] * np.exp(s[0] - m2ms) for s in m2)
    for i, (m, ll, a, b) in enumerate(m2):
        emit += [format_vector_definition("uint32_t", f"moment2_beat{i}_in", pack_attn(m, ll, [a, b]),
                                          alignment=64, hex_bits=32, cast_hex=True)]
    emit += [format_scalar_definition("uint32_t", "moment2_m_golden", u32(m2ms))]
    emit += [format_scalar_definition("uint32_t", "moment2_l_golden", u32(m2l))]
    emit += [format_scalar_definition("uint32_t", "moment2_a_golden", u32(m2a))]
    emit += [format_scalar_definition("uint32_t", "moment2_b_golden", u32(m2b))]

    # ---- ATTN: two shard partials (m, ℓ, O) -> flash-attention merged triple ----
    shards = []
    for _ in range(2):
        m = float(rng.uniform(-2.0, 5.0))
        l = float(rng.uniform(1.0, 5.0))
        o = rng.uniform(-3.0, 3.0, size=DHEAD).astype(np.float32).astype(np.float64)
        shards.append((m, l, o))
    ams = max(s[0] for s in shards)
    al = sum(s[1] * np.exp(s[0] - ams) for s in shards)
    ao = np.sum([s[2] * np.exp(s[0] - ams) for s in shards], axis=0)
    for i, (m, l, o) in enumerate(shards):
        emit += [format_vector_definition("uint32_t", f"attn_beat{i}_in", pack_attn(m, l, o),
                                          alignment=64, hex_bits=32, cast_hex=True)]
    emit += [format_scalar_definition("uint32_t", "attn_m_golden", u32(ams))]
    emit += [format_scalar_definition("uint32_t", "attn_l_golden", u32(al))]
    emit += [format_vector_definition("uint32_t", "attn_o_golden",
                                      np.array([u32(x) for x in ao], dtype=np.uint32),
                                      alignment=64, hex_bits=32, cast_hex=True)]
    return "\n\n".join(emit)


def main():
    parser = argparse.ArgumentParser(
        description="Data for the local-loopback UnifiedMonoidMergeRt smoke test")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
