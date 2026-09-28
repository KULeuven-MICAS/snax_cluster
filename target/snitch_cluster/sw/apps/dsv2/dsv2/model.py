# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V2-Lite's constants, read from its config.json, and its YaRN RoPE.

config.json is a verbatim copy of the one in huggingface.co/deepseek-ai/DeepSeek-V2-Lite.
Everything derived from it follows that repository's modeling_deepseek.py:

    softmax scale   q_head_dim^-0.5 * mscale^2, mscale = 0.1 * mscale_all_dim * ln(factor) + 1
    YaRN RoPE       DeepseekV2YarnRotaryEmbedding: per-pair frequencies ramped between the
                    extrapolated and the interpolated ones; the cos/sin magnitude factor is
                    mscale / mscale_all_dim = 1 for this config

RoPE PAIR ORDER. modeling_deepseek.apply_rotary_pos_emb de-interleaves a 64-wide q_pe /
k_pe into [even | odd] and then rotates halves, so it rotates the ADJACENT pairs
(x[2i], x[2i+1]) by angle pos * f_i and stores the result de-interleaved. The device
rotates the same pairs in place and keeps them interleaved (rope_interleaved below).
Scores are identical, since q_pe and k_pe move together; only the stored order of k_pe
differs from HF's, and the golden uses the device's order throughout.
"""

import json
import math
import os
from dataclasses import dataclass

import numpy as np

CONFIG_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


@dataclass(frozen=True)
class Dims:
    hidden: int          # d_model, 2048
    heads: int           # 16
    q_nope: int          # 128
    q_rope: int          # 64
    kv_rank: int         # 512, the latent c
    v_head: int          # 128
    moe_inter: int       # 1408, one expert's hidden size
    n_routed: int        # 64
    top_k: int           # 6
    n_shared: int        # 2
    norm_topk: bool      # False: the top-6 weights are not renormalised
    routed_scale: float  # 1.0
    rms_eps: float
    rope_theta: float
    rope_factor: float
    rope_orig_max: int
    beta_fast: float
    beta_slow: float
    mscale: float
    mscale_all_dim: float

    @property
    def q_head(self):
        return self.q_nope + self.q_rope

    @property
    def latent(self):
        """One cached row: [c | k_pe], 576."""
        return self.kv_rank + self.q_rope

    @property
    def shared_inter(self):
        return self.moe_inter * self.n_shared

    @property
    def softmax_scale(self):
        scale = self.q_head ** -0.5
        if self.mscale_all_dim:
            m = yarn_get_mscale(self.rope_factor, self.mscale_all_dim)
            scale *= m * m
        return scale


def yarn_get_mscale(scale=1.0, mscale=1.0):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def load(path=CONFIG_JSON):
    with open(path) as f:
        c = json.load(f)
    if c.get("q_lora_rank") is not None:
        raise ValueError("this golden models the V2-Lite query path, which has no q_lora_rank")
    rs = c["rope_scaling"]
    if rs.get("type") != "yarn":
        raise ValueError(f"RoPE scaling {rs.get('type')!r} is not modelled")
    return Dims(
        hidden=c["hidden_size"], heads=c["num_attention_heads"],
        q_nope=c["qk_nope_head_dim"], q_rope=c["qk_rope_head_dim"],
        kv_rank=c["kv_lora_rank"], v_head=c["v_head_dim"],
        moe_inter=c["moe_intermediate_size"], n_routed=c["n_routed_experts"],
        top_k=c["num_experts_per_tok"], n_shared=c["n_shared_experts"],
        norm_topk=bool(c["norm_topk_prob"]), routed_scale=float(c["routed_scaling_factor"]),
        rms_eps=float(c["rms_norm_eps"]), rope_theta=float(c["rope_theta"]),
        rope_factor=float(rs["factor"]), rope_orig_max=int(rs["original_max_position_embeddings"]),
        beta_fast=float(rs["beta_fast"]), beta_slow=float(rs["beta_slow"]),
        mscale=float(rs["mscale"]), mscale_all_dim=float(rs["mscale_all_dim"]),
    )


def _yarn_correction_dim(num_rotations, dim, base, max_pos):
    return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def yarn_inv_freq(d: Dims):
    """The q_rope/2 rotation frequencies, one per adjacent pair, as modeling_deepseek builds them."""
    dim, base = d.q_rope, d.rope_theta
    exps = np.arange(0, dim, 2, dtype=np.float64) / dim
    freq_extra = 1.0 / (base ** exps)
    freq_inter = 1.0 / (d.rope_factor * base ** exps)
    low = max(math.floor(_yarn_correction_dim(d.beta_fast, dim, base, d.rope_orig_max)), 0)
    high = min(math.ceil(_yarn_correction_dim(d.beta_slow, dim, base, d.rope_orig_max)), dim - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2, dtype=np.float64) - low) / (high - low), 0.0, 1.0)
    mask = 1.0 - ramp
    return freq_inter * (1.0 - mask) + freq_extra * mask


def rope_cos_sin(d: Dims, pos):
    """cos and sin of every pair's angle at position `pos`, float64, q_rope/2 each.

    The magnitude factor mscale(factor, mscale) / mscale(factor, mscale_all_dim) is 1 for
    V2-Lite; it is applied anyway so a config where it is not stays correct.
    """
    ang = float(pos) * yarn_inv_freq(d)
    amp = yarn_get_mscale(d.rope_factor, d.mscale) / yarn_get_mscale(d.rope_factor, d.mscale_all_dim)
    return np.cos(ang) * amp, np.sin(ang) * amp


def rope_interleaved(x, cos, sin):
    """Rotate adjacent pairs of x [..., 2P] in place order: the device's (and the golden's) RoPE."""
    x = np.asarray(x)
    xe, xo = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = xe * cos - xo * sin
    out[..., 1::2] = xo * cos + xe * sin
    return out


def rope_hf(x, cos, sin):
    """modeling_deepseek.apply_rotary_pos_emb on one vector set: de-interleave, rotate halves."""
    x = np.asarray(x)
    P = x.shape[-1] // 2
    xd = np.concatenate([x[..., 0::2], x[..., 1::2]], axis=-1)
    c2, s2 = np.concatenate([cos, cos]), np.concatenate([sin, sin])
    rot = np.concatenate([-xd[..., P:], xd[..., :P]], axis=-1)
    return xd * c2 + rot * s2


def hf_to_interleaved(y):
    """HF's de-interleaved RoPE output -> the device's interleaved order."""
    y = np.asarray(y)
    P = y.shape[-1] // 2
    out = np.empty_like(y)
    out[..., 0::2] = y[..., :P]
    out[..., 1::2] = y[..., P:]
    return out
