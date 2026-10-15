# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Operand layouts of VersaCore's array shape 3, (8, 8, 16), with INT4 weights and a row-major A
(include/snax-dsv2-g3.h). Shared by the datagens of snax-dsv2-gemm3-bench and
snax-dsv2-moe-prefill."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))  # the util package
from util.fp import bits16, d_port  # noqa E402
from util.layout import pack_int4, to_a, to_b, to_d  # noqa E402

MESH3 = (8, 8, 16)  # (Mu, Ku, Nu) of array shape 3
LAYOUTS = {"b": 1, "paired": 2, "parity": 3}  # g3_lay_t
ALAYOUTS = {"row": 0, "blocked": 1, "ilv": 2}  # g3_alay_t
PARITY_OFF = 8      # a parity-layout chunk starts one word into its slab


def check_hwcfg(hw):
    """Raise unless array shape 3 of the cluster cfg is (8, 8, 16)."""
    acc = hw["snax_versacore_core_template"]["snax_acc_cfg"][0]
    shapes = acc["snax_versacore_spatial_unrolling"][0]
    if len(shapes) <= 3 or tuple(int(v) for v in shapes[3]) != MESH3:
        raise ValueError(f"array shape 3 of the cfg is not {MESH3}: {shapes}")


def pitch(nbytes):
    """A row-major operand's row pitch: a row and 16 bytes (a quantiser writes 16 values past
    the end), beat-aligned, and 32 mod 64 so the eight rows of a pass sit in eight banks."""
    p = int(nbytes) + 16
    p += (32 - p % 64) % 64
    return p


def a_rows(X, row_pitch):
    """X [M, K] INT8 as M rows of row_pitch bytes (the row's K values, then zeros)."""
    M, K = X.shape
    out = np.zeros((M, row_pitch), dtype=np.int8)
    out[:, :K] = X
    return out.reshape(-1)


def a_blob(X, alay, row_pitch):
    """X [M, K] INT8 as the bytes of A layout `alay` ('row': M rows of row_pitch bytes; 'blocked':
    A-layout tiles of 8 x 8, pass p of row tile m at 64 (kt m + p); 'ilv': the passes of each row
    tile in two interleaved K halves, pass p at 128 (p % H) + 64 (p // H))."""
    if alay == "row":
        return a_rows(X, row_pitch)
    M, K = X.shape
    a = np.asarray(to_a(X, MESH3)).reshape(M // MESH3[0], K // MESH3[1], 64)
    if alay == "blocked":
        return np.ascontiguousarray(a).reshape(-1)
    kt = K // MESH3[1]
    if kt % 2:
        raise ValueError("the interleaved A takes an even number of passes of K")
    h = kt // 2
    out = np.empty_like(a)
    for p in range(kt):
        out[:, 2 * (p % h) + (p // h), :] = a[:, p, :]
    return np.ascontiguousarray(out).reshape(-1)


def b_blob(W, lay):
    """W [K, N] INT4 values (int8 in [-8, 7]) as the packed bytes of layout `lay` ('b', 'paired',
    'parity'), the first byte of the operand (a parity chunk's slab offset not included)."""
    K, N = W.shape
    if lay == "b":
        return pack_int4(to_b(W, MESH3))
    if lay == "paired":
        if N % (2 * MESH3[2]):
            raise ValueError("the paired layout takes column tiles in pairs")
        return pack_int4(to_b(W, (MESH3[0], MESH3[1], 2 * MESH3[2])))
    if lay == "parity":
        if (K // MESH3[1]) % 2:
            raise ValueError("the parity layout takes an even number of passes of K")
        b = pack_int4(to_b(W, MESH3))
        return np.ascontiguousarray(
            np.asarray(b).reshape(N // MESH3[2], K // MESH3[1] // 2, 2, 8, 8)
            .transpose(0, 1, 3, 2, 4)).reshape(-1)
    raise ValueError(lay)


def d_blob(X, W, k):
    """The expected D = RNE(X . W * 2^-k) as FP16 bits in the D-layout (tile (m, n) 256 bytes)."""
    acc = X.astype(np.int64) @ W.astype(np.int64)
    y = d_port(acc, k)
    if np.any(np.isinf(y.astype(np.float32))):
        raise ValueError(f"an output overflows FP16 at k = {k}")
    return bits16(to_d(y, MESH3))
