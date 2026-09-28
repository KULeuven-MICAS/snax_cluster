# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Golden data for DeepSeek-V2-Lite layer 1 on snax_split_cluster.

One golden drives all three test levels (this repository's apps, HeMAiA on one cluster,
HeMAiA on four clusters):

    model.py      the model's constants, read from its config.json, and its YaRN RoPE
    weights.py    seeded random BF16 weights at the real shapes
    reference.py  layer 1 in float32, following modeling_deepseek.py
    hwmodel.py    layer 1 as the cluster computes it: every stage bit-exact
    simd.py       the SIMD's FP32 arithmetic: its fused multiply-add, tables and reduce order
    calib.py      static activation scales from sample tokens
    pack.py       the weights as the device reads them: INT8 per column, B-layout
    golden.py     one pack per (seed, L), further passes after it, and the per-stage report
    fp.py         the bit-level numerics every stage shares
    layout.py     the array's operand layouts
    appdata.py    what the block kernels read from an app's data.h

The apps' data generators, next to this package, put its parent directory on sys.path and
import from here.
"""
