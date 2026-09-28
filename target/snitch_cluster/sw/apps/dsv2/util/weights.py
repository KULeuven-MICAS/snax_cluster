# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Layer-1 weights at DeepSeek-V2-Lite's real shapes: seeded random, stored as BF16 values.

Every matrix is in [in, out] orientation, so a projection is x @ W (the checkpoint's
nn.Linear weights are [out, in]: W = weight.T). The names follow modeling_deepseek.py:

    in_norm   [2048]             input_layernorm gain
    wq        [2048, 3072]       q_proj: per head [q_nope 128 | q_pe 64]
    wdkv      [2048, 576]        kv_a_proj_with_mqa: [c 512 | k_pe 64]
    kv_norm   [512]              kv_a_layernorm gain
    wkvb      [512, 4096]        kv_b_proj: per head [k_nope 128 | v 128]
    wo        [2048, 2048]       o_proj, input = the 16 heads' 128-wide outputs
    post_norm [2048]             post_attention_layernorm gain
    wr        [2048, 64]         the MoE gate (router)
    shared    gate, up [2048, 2816], down [2816, 2048]
    expert(e) gate, up [2048, 1408], down [1408, 2048]

The 64 routed experts hold 553 M parameters, so they are never materialised together:
expert(e) draws expert e from its own stream, and the same (seed, e) always yields the
same expert. Magnitudes follow the checkpoint's initialiser (std 0.02) and gains are
drawn around 1, so the norm-gain folding is exercised rather than trivially the identity.
"""

import numpy as np

from .fp import bf16

INIT_STD = 0.02


class LayerWeights:
    def __init__(self, dims, seed):
        self.d = dims
        self.seed = int(seed)
        rng = np.random.default_rng([self.seed, 1])
        d = dims
        H = d.heads

        def mat(k, n):
            return bf16(rng.normal(0.0, INIT_STD, size=(k, n)).astype(np.float32))

        def gain(n):
            return bf16(rng.uniform(0.5, 1.5, size=n).astype(np.float32))

        self.in_norm = gain(d.hidden)
        self.wq = mat(d.hidden, H * d.q_head)
        self.wdkv = mat(d.hidden, d.kv_rank + d.q_rope)
        self.kv_norm = gain(d.kv_rank)
        self.wkvb = mat(d.kv_rank, H * (d.q_nope + d.v_head))
        self.wo = mat(H * d.v_head, d.hidden)
        self.post_norm = gain(d.hidden)
        self.wr = mat(d.hidden, d.n_routed)
        self.shared = {"gate": mat(d.hidden, d.shared_inter), "up": mat(d.hidden, d.shared_inter),
                       "down": mat(d.shared_inter, d.hidden)}
        self._experts = {}

    def expert(self, e):
        if not 0 <= e < self.d.n_routed:
            raise IndexError(f"expert {e} does not exist")
        if e not in self._experts:
            rng = np.random.default_rng([self.seed, 1000 + e])
            d = self.d

            def mat(k, n):
                return bf16(rng.normal(0.0, INIT_STD, size=(k, n)).astype(np.float32))

            self._experts[e] = {"gate": mat(d.hidden, d.moe_inter), "up": mat(d.hidden, d.moe_inter),
                                "down": mat(d.moe_inter, d.hidden)}
        return self._experts[e]

    # ---- the per-head views the absorbed attention uses --------------------------------
    def w_uk(self, h):
        """[512, 128]: c -> k_nope of head h (the first half of head h's kv_b columns)."""
        d = self.d
        base = h * (d.q_nope + d.v_head)
        return self.wkvb[:, base:base + d.q_nope]

    def w_uv(self, h):
        """[512, 128]: c -> v of head h (the second half of head h's kv_b columns)."""
        d = self.d
        base = h * (d.q_nope + d.v_head) + d.q_nope
        return self.wkvb[:, base:base + d.v_head]
