# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V2-Lite layer 1 in float32: one new token at position L, L tokens cached.

This follows modeling_deepseek.py (DeepseekV2DecoderLayer with DeepseekV2Attention and
DeepseekV2MoE, eval mode) operation for operation, in float32 from the BF16 weights:

    h   = x + o_proj(attn(input_layernorm(x)))
    out = h + moe(post_attention_layernorm(h))

Attention is computed the way HF computes it -- EXPANDED: every cached latent is
up-projected to per-head keys and values. The ABSORBED form the device runs (the query
mapped into the latent space, the weighted sum taken over latents, W_UV applied once) is
computed alongside as `q_lat` / `o_lat`, and the two must agree to float32 rounding;
test_dsv2.py checks that.

THE CACHE. HF caches expanded keys and values; the absorbed form needs only, per token,
the normalised latent and the rotated RoPE key. `Cache` holds exactly that:

    c_unit [T, 512]   the latent after the RMS normalisation, WITHOUT the kv_a_layernorm
                      gain (the device folds the gain into W_UK and W_UV)
    kpe    [T, 64]    k_pe rotated at its own position, in the interleaved pair order

and `cache_entry` computes one row from a token's hidden state, the way the layer does
when it appends.
"""

from dataclasses import dataclass

import numpy as np

from .model import rope_cos_sin, rope_interleaved

F32 = np.float32


def rmsnorm_unit(x, eps):
    """x * rsqrt(mean(x^2) + eps) in float32 -- DeepseekV2RMSNorm without its gain."""
    x = np.asarray(x, dtype=F32)
    var = np.mean(x * x, axis=-1, keepdims=True, dtype=F32)
    return (x / np.sqrt(var + F32(eps))).astype(F32)


def rmsnorm(x, g, eps):
    return (g * rmsnorm_unit(x, eps)).astype(F32)


def silu(x):
    x = np.asarray(x, dtype=F32)
    return (x / (F32(1.0) + np.exp(-x))).astype(F32)


def softmax(s, axis=-1):
    s = np.asarray(s, dtype=F32)
    e = np.exp(s - s.max(axis=axis, keepdims=True))
    return (e / e.sum(axis=axis, keepdims=True, dtype=F32)).astype(F32)


def swiglu_mlp(x, w):
    """DeepseekV2MLP: down(silu(gate(x)) * up(x)), returning the hidden activation too."""
    a = (silu(x @ w["gate"]) * (x @ w["up"])).astype(F32)
    return (a @ w["down"]).astype(F32), a


def top_k(p, k):
    """Indices of the k largest, largest first; equal values go to the lower index."""
    order = np.lexsort((np.arange(p.shape[-1]), -np.asarray(p, dtype=np.float64)))
    return order[:k]


@dataclass
class Cache:
    c_unit: np.ndarray   # [T, kv_rank] float32
    kpe: np.ndarray      # [T, q_rope]  float32, interleaved pair order

    @property
    def length(self):
        return self.c_unit.shape[0]


def cache_entry(W, x, pos):
    """The cache row the layer appends for a token with hidden state x at position pos."""
    d = W.d
    xn = rmsnorm(x, W.in_norm, d.rms_eps)
    ckv = (xn @ W.wdkv).astype(F32)
    c_unit = rmsnorm_unit(ckv[..., :d.kv_rank], d.rms_eps)
    cos, sin = rope_cos_sin(d, pos)
    kpe = rope_interleaved(ckv[..., d.kv_rank:].astype(np.float64), cos, sin).astype(F32)
    return c_unit, kpe, ckv


def build_cache(W, xs, positions):
    """Cache rows for previous tokens with hidden states xs [T, 2048] at `positions`."""
    rows = [cache_entry(W, x, p) for x, p in zip(xs, positions)]
    return Cache(np.stack([r[0] for r in rows]).astype(F32), np.stack([r[1] for r in rows]).astype(F32))


def attention(W, xn, pos, cache):
    """MLA for one token (xn = the normed hidden state) against cache + itself. Returns a dict."""
    d = W.d
    H = d.heads
    q = (xn @ W.wq).astype(F32).reshape(H, d.q_head)
    q_nope, q_pe = q[:, :d.q_nope], q[:, d.q_nope:]
    cos, sin = rope_cos_sin(d, pos)
    q_pe = rope_interleaved(q_pe.astype(np.float64), cos, sin).astype(F32)

    ckv = (xn @ W.wdkv).astype(F32)
    c_new = rmsnorm_unit(ckv[:d.kv_rank], d.rms_eps)
    kpe_new = rope_interleaved(ckv[d.kv_rank:].astype(np.float64), cos, sin).astype(F32)
    c_unit = np.concatenate([cache.c_unit, c_new[None]], axis=0)          # [T+1, 512]
    kpe = np.concatenate([cache.kpe, kpe_new[None]], axis=0)              # [T+1, 64]
    c_norm = (c_unit * W.kv_norm).astype(F32)

    # ---- expanded, as HF: per-head keys and values from every cached latent -------------
    scores = np.empty((H, c_unit.shape[0]), dtype=F32)
    values = []
    for h in range(H):
        k_nope = (c_norm @ W.w_uk(h)).astype(F32)                          # [T+1, 128]
        values.append((c_norm @ W.w_uv(h)).astype(F32))                    # [T+1, 128]
        scores[h] = (k_nope @ q_nope[h] + kpe @ q_pe[h]).astype(F32) * F32(d.softmax_scale)
    p = softmax(scores)
    o = np.stack([(p[h] @ values[h]).astype(F32) for h in range(H)])     # [H, 128]
    attn_out = (o.reshape(-1) @ W.wo).astype(F32)

    # ---- absorbed: the same quantities through the latent space -------------------------
    q_lat = np.stack([(W.w_uk(h) @ q_nope[h]).astype(F32) for h in range(H)])   # [H, 512]
    s_lat = ((q_lat @ c_norm.T) + (q_pe @ kpe.T)).astype(F32) * F32(d.softmax_scale)
    o_lat = (softmax(s_lat) @ c_norm).astype(F32)                                # [H, 512]

    return dict(q=q, q_nope=q_nope, q_pe=q_pe, ckv=ckv, c_new=c_new, kpe_new=kpe_new,
                scores=scores, p=p, o=o, attn_out=attn_out, q_lat=q_lat, s_lat=s_lat,
                o_lat=o_lat)


def moe(W, hn):
    """DeepseekV2MoE for one token (hn = the post-attention-normed state). Returns a dict."""
    d = W.d
    logits = (hn @ W.wr).astype(F32)
    probs = softmax(logits)
    ids = top_k(probs, d.top_k)
    w = probs[ids].astype(F32)
    if d.top_k > 1 and d.norm_topk:
        w = (w / (w.sum() + F32(1e-20))).astype(F32)
    else:
        w = (w * F32(d.routed_scale)).astype(F32)
    experts = []
    acts = []
    for e in ids:
        y, a = swiglu_mlp(hn, W.expert(int(e)))
        experts.append(y)
        acts.append(a)
    shared, shared_act = swiglu_mlp(hn, W.shared)
    routed = np.zeros(d.hidden, dtype=F32)
    for wi, y in zip(w, experts):
        routed = (routed + wi * y).astype(F32)
    return dict(logits=logits, probs=probs, ids=ids, w=w, experts=np.stack(experts),
                expert_acts=np.stack(acts), shared=shared, shared_act=shared_act,
                out=(routed + shared).astype(F32))


def layer(W, x, pos, cache):
    """Layer 1 for one token: every intermediate, keyed by stage."""
    d = W.d
    x = np.asarray(x, dtype=F32)
    xn = rmsnorm(x, W.in_norm, d.rms_eps)
    att = attention(W, xn, pos, cache)
    h = (x + att["attn_out"]).astype(F32)
    hn = rmsnorm(h, W.post_norm, d.rms_eps)
    mo = moe(W, hn)
    out = (h + mo["out"]).astype(F32)
    return dict(x=x, xn=xn, h=h, hn=hn, out=out, **{f"att_{k}": v for k, v in att.items()},
                **{f"moe_{k}": v for k, v in mo.items()})
