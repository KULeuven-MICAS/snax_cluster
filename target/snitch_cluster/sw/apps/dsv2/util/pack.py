# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Layer-1 weights as the device reads them: INT8, one scale per output column, B-layout.

QUANTISATION. Symmetric INT8 per output column: s[n] = max_k |W'[k, n]| / 127 and
q[k, n] = sat_127(rne(W'[k, n] / s[n])), so W' ~= q * s[None, :]. W' is the weight with
the RMSNorm gain that precedes it folded in, which is what lets the device norm run
without a gain:

    input_layernorm            -> the rows of W_Q and W_DKV
    kv_a_layernorm (latent)    -> the absorbed W_UK (its latent columns) and W_UV (rows)
    post_attention_layernorm   -> the rows of the router and of every expert's gate and up

LAYOUT. Each matrix is stored whole in the array's B-layout. B-layout is n-major -- block
(n, k) of Nu columns x Ku rows at ((n*K_T + k)*Nu + c)*Ku + s -- so the first K*C bytes
are exactly the B-layout of columns [0, C), the next K*C those of [C, 2C), and so on. A
GEMV streams the blob in chunks of C = 64 columns and each chunk is itself a complete B
operand, with no repacking. layout.b_chunks() spells the chunks out and must agree.

FUSED MATRICES. An expert's gate and up form ONE [2048, 2*1408] matrix, gate columns
first, so one streamed GEMV produces both halves the SwiGLU reads. The two shared
experts are one MLP of hidden size 2816 in the checkpoint and fuse the same way.

ABSORBED ATTENTION. Per head h, with g the latent gain:
    W_UK operand  [128, 512]: B[k, n] = W_UK_h[n, k] * g[n]   (q~_h = q_nope_h @ B)
    W_UV operand  [512, 128]: B[k, n] = g[k] * W_UV_h[k, n]   (o_h = o~_h @ B)
stored as 16 consecutive per-head B-layout blocks.
"""

from dataclasses import dataclass

import numpy as np

from .layout import MESH, from_b, to_b

F32 = np.float32


@dataclass
class Packed:
    """One quantised weight: q [K, N] int8 and its per-column scale s [N] float32."""
    name: str
    q: np.ndarray
    s: np.ndarray

    @property
    def K(self):
        return self.q.shape[0]

    @property
    def N(self):
        return self.q.shape[1]

    def blob(self, mesh=MESH):
        """The bytes the device streams: the whole matrix in B-layout."""
        return to_b(self.q, mesh)

    def dequant(self):
        return (self.q.astype(F32) * self.s[None, :]).astype(F32)


def quantize_columns(W, row_gain=None, col_gain=None):
    """Symmetric per-output-column INT8 of W [K, N] with gains folded into rows / columns."""
    Wf = np.asarray(W, dtype=F32)
    if row_gain is not None:
        Wf = Wf * np.asarray(row_gain, dtype=F32)[:, None]
    if col_gain is not None:
        Wf = Wf * np.asarray(col_gain, dtype=F32)[None, :]
    amax = np.max(np.abs(Wf), axis=0)
    s = np.where(amax > 0, amax / F32(127.0), F32(1.0)).astype(F32)
    q = np.clip(np.rint(Wf / s[None, :]), -127, 127).astype(np.int8)
    return q, s


def pack(name, W, row_gain=None, col_gain=None):
    q, s = quantize_columns(W, row_gain, col_gain)
    return Packed(name, q, s)


def unpack(blob, K, N, mesh=MESH):
    """Inverse of Packed.blob(): the [K, N] int8 matrix."""
    return from_b(np.asarray(blob, dtype=np.int8), K, N, mesh)


@dataclass
class PackedHeads:
    """16 per-head operands of one absorbed projection, [H, K, N] int8 + [H, N] scales."""
    name: str
    q: np.ndarray
    s: np.ndarray

    def blob(self, mesh=MESH):
        return np.concatenate([to_b(self.q[h], mesh) for h in range(self.q.shape[0])])

    def dequant(self):
        return (self.q.astype(F32) * self.s[:, None, :]).astype(F32)


def pack_heads(name, mats, row_gain=None, col_gain=None):
    qs, ss = zip(*(quantize_columns(m, row_gain, col_gain) for m in mats))
    return PackedHeads(name, np.stack(qs), np.stack(ss))


class LayerPack:
    """Every layer-1 weight, quantised and laid out; routed experts packed on demand."""

    def __init__(self, W):
        self.W = W
        d = W.d
        self.wq = pack("wq", W.wq, row_gain=W.in_norm)
        self.wdkv = pack("wdkv", W.wdkv, row_gain=W.in_norm)
        self.wuk = pack_heads("wuk", [W.w_uk(h).T for h in range(d.heads)], col_gain=W.kv_norm)
        self.wuv = pack_heads("wuv", [W.w_uv(h) for h in range(d.heads)], row_gain=W.kv_norm)
        self.wo = pack("wo", W.wo)
        self.wr = pack("wr", W.wr, row_gain=W.post_norm)
        sh = W.shared
        self.shared_gu = pack("shared_gu", np.concatenate([sh["gate"], sh["up"]], axis=1),
                              row_gain=W.post_norm)
        self.shared_down = pack("shared_down", sh["down"])
        self._experts = {}

    def expert(self, e):
        """(gate_up [2048, 2816], down [1408, 2048]) of routed expert e."""
        if e not in self._experts:
            w = self.W.expert(e)
            self._experts[e] = (
                pack(f"e{e}_gu", np.concatenate([w["gate"], w["up"]], axis=1),
                     row_gain=self.W.post_norm),
                pack(f"e{e}_down", w["down"]))
        return self._experts[e]
