# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Layer 1 as snax_split_cluster computes it, stage by stage.

Every stage takes the device's own output of the stage before (the CHAINED view) and
works at the device's precision:

    GEMM stages     INT8 x INT8 -> exact INT32 -> D port RNE(acc * 2^-k) to FP16.
                    Bit-exact.
    dequantise      y (.) s with s[n] = s_x * s_w[n] * 2^k rounded to FP16: one SIMD EW MUL,
                    FP32 product, RNE to FP16. Bit-exact.
    quantise        Fp16ToInt8 with one static FP32 inv_scale per tensor. Bit-exact.
    SIMD chains     each stage widens FP16 to FP32, computes, narrows to FP16. Bit-exact: the
                    exp / silu / rsqrt tables, the map's fused multiply-add and the reduce's
                    order of summation are simd.py's models of the RTL.

The data a kernel test needs -- a stage's inputs, its expected outputs, its scales and
its D-port shift -- are all in the dict run() returns; golden.py packages them.

THE ATTENTION TILE follows snax-flashattn-decode's online softmax, transposed: scores
S^T = K . Q^T are [key][query], so a SIMD lane is a query and the row max over keys is a
lanewise max. The query tile has Br = 32 lanes: the 16 heads and 16 zero rows (the SIMD
beat is 32 FP16 lanes). Per KV tile j of Bc keys:

    S16   = RNE(S * 2^-k_s)                         D port, k_s from depth 576
    m     = max(m_old, max over keys of S16)         exact
    corr  = exp(a' * fp16(m_old - m))                EW0 ADD -> Map EXP
    P     = exp(a' * fp16(S16 - m))                  EW0 ADD -> Map EXP, FP16
    l     = fp16(fp16(sum_keys P) + fp16(corr * l))  Reduce ADD, EW1 MUL
    P8    = sat(rne(127 * P))                        Fp16ToInt8
    O     = rne(O * corr[query]) + V^T . P8^T        Int32ColumnScale on C, then the GEMM

with a' = softmax_scale * s_q * s_c * 2^k_s. After the last tile O^T leaves through the
D port as FP16 with shift k_o (depth L+1), and a SIMD MUL by c / l per query, c = 2^k_o
s_c / 127, gives o~ = sum_t p_t c_t in the latent's unit domain.
"""

from dataclasses import dataclass, field

import numpy as np

from . import simd
from .fp import F16, F32, F64, add16, d_port, d_shift_for_depth, mul16, quant_i8, to16
from .model import rope_cos_sin

P8_SCALE = 127.0


# ---- one-stage device operators ------------------------------------------------------------

def rmsnorm(x16, n=None):
    """RMSNorm of a row without a gain, as dsv2_rmsnorm_row runs it: Reduce SUMSQ over the row
    (FP16 out), Map RSQRT with a = 1/n, EW MUL. n defaults to the row length; it must be a power
    of two for 1/n to be exact. A 2-D input is normalised row by row."""
    x = np.asarray(x16, dtype=F16)
    if x.ndim > 1:
        return np.stack([rmsnorm(r, n) for r in x])
    n = x.shape[-1] if n is None else n
    ssq = simd.reduce_row(simd.as_beats(x), simd.SUMSQ)
    if not np.isfinite(ssq):
        raise FloatingPointError("a row's sum of squares overflows the FP16 reduce output")
    inv = simd.stream_map(ssq, F32(1.0 / n), 0.0, simd.RSQRT)
    return mul16(x, inv)


def exp16(x16, a=1.0):
    """Map EXP: exp(a * x) on FP16 lanes."""
    return simd.stream_map(x16, a, 0.0, simd.EXP)


def silu16(x16):
    """Map SILU on FP16 lanes."""
    return simd.stream_map(x16, 1.0, 0.0, simd.SILU)


def rope(x16, cos16, sin16):
    """Adjacent-pair RoPE as three elementwise stages: x*cos, swap(x)*sin_signed, add.

    cos16 / sin16 are the per-pair values (P each). The tables the device multiplies are
    cos repeated per pair and sin with the sign folded in: [-s0, +s0, -s1, +s1, ...].
    """
    x = np.asarray(x16, dtype=F16)
    c_rep = np.repeat(np.asarray(cos16, dtype=F16), 2)
    s_sgn = np.empty(x.shape[-1], dtype=F16)
    s_sgn[0::2] = -np.asarray(sin16, dtype=F16)
    s_sgn[1::2] = np.asarray(sin16, dtype=F16)
    xs = np.empty_like(x)
    xs[..., 0::2] = x[..., 1::2]
    xs[..., 1::2] = x[..., 0::2]
    return add16(mul16(x, c_rep), mul16(xs, s_sgn))


def gemv(xq, Wq, k):
    """One row through the array: exact INT32 dot products, then the D port at shift k."""
    acc = np.asarray(xq, dtype=np.int64) @ np.asarray(Wq, dtype=np.int64)
    return d_port(acc, k), acc


def dequant_scale(s_x, s_w, k):
    """The per-column dequantisation factor s_x * s_w[n] * 2^k, as the FP16 the SIMD reads."""
    return (F64(s_x) * np.asarray(s_w, dtype=F64) * F64(2.0 ** k)).astype(F16)


def softmax16(x16):
    """One-row FP16 softmax on the SIMD (dsv2_softmax_row): the row max; EW0 ADD -m -> Map EXP ->
    Reduce ADD (the sum); the reciprocal as RSQRT(s * s) (EW0 MUL, then the map), exact for any
    positive s whose square fits FP16; then MUL."""
    x = np.asarray(x16, dtype=F16)
    m = simd.reduce_row(simd.as_beats(x), simd.MAX)
    e = exp16(add16(x, -m))
    s = simd.reduce_row(simd.as_beats(e), simd.ADD)
    r = simd.stream_map(mul16(s, s), 1.0, 0.0, simd.RSQRT)
    return mul16(e, r)


def top_k16(p16, k):
    """Top-k over FP16 bit patterns (non-negative here), ties to the lower index."""
    u = np.asarray(p16, dtype=F16).view(np.uint16).astype(np.int64)
    order = np.lexsort((np.arange(u.shape[0]), -u))
    return order[:k]


# ---- static scales ---------------------------------------------------------------------------

@dataclass
class QScale:
    """One tensor's static INT8 scale: inv (the FP32 value the quantiser's CSR holds)."""
    inv: float

    @property
    def s(self):
        return 1.0 / float(F32(self.inv))

    @classmethod
    def for_amax(cls, amax):
        return cls(float(F32(127.0 / max(float(amax), 1e-30))))


@dataclass
class Scales:
    x: QScale          # the normed layer input, feeding W_Q and W_DKV
    qn: QScale         # q_nope, all heads, feeding W_UK
    c: QScale          # the normed latent: cache (key and value copies)
    kpe: QScale        # the rotated k_pe: cache
    qt: QScale         # q~, the absorbed query; tied to qpe by s_qt * s_c = s_qpe * s_kpe
    qpe: QScale        # the rotated q_pe
    ot: QScale         # o~, the attention output in the latent domain, feeding W_UV
    o: QScale          # the 16 heads' outputs, feeding W_O
    h: QScale          # the post-attention normed state, feeding the router and experts
    shared_a: QScale   # the shared experts' SwiGLU output
    expert_a: dict = field(default_factory=dict)   # routed expert id -> SwiGLU output scale


# ---- the attention tile ----------------------------------------------------------------------

MASKED_SCORE = -65504.0  # what the kernel writes over a padding key's score


def mla_attention(Q8, Kc, Vc, a_exp, k_s, bc, masked=()):
    """Online softmax over key tiles of `bc`. Q8 [Br, 576], Kc [T, 576], Vc [T, 512] int8.

    A last tile with fewer than bc keys runs whole: the cache's rows past T are zero, and the
    kernel overwrites their scores with -65504 (one Map task) before the softmax, so their P,
    and with it their weight in l and O, is exactly 0. `masked` lists (key, query rows) a query
    must not see -- a later token of the same pass -- and gets the same -65504, written by a
    Map task whose writer enables only those rows' lanes.

    Returns the per-tile state and the final INT32 O^T [512, Br], m and l per query lane.
    """
    T = Kc.shape[0]
    pad = -T % bc
    Kc = np.concatenate([Kc, np.zeros((pad, Kc.shape[1]), dtype=Kc.dtype)])
    Vc = np.concatenate([Vc, np.zeros((pad, Vc.shape[1]), dtype=Vc.dtype)])
    Br = Q8.shape[0]
    a32 = F32(a_exp)
    m16 = np.full(Br, -65504.0, dtype=F16)
    l16 = np.zeros(Br, dtype=F16)
    o = np.zeros((Vc.shape[1], Br), dtype=np.int64)
    tiles = []
    nt = Kc.shape[0] // bc
    for j in range(nt):
        K = Kc[j * bc:(j + 1) * bc].astype(np.int64)
        V = Vc[j * bc:(j + 1) * bc].astype(np.int64)
        S = K @ Q8.astype(np.int64).T                                   # [bc, Br], S^T
        S16 = d_port(S, k_s)
        if not np.all(np.isfinite(S16)):
            raise FloatingPointError("a score overflows FP16 after the D-port shift")
        if j == nt - 1 and pad:
            S16[bc - pad:, :] = F16(MASKED_SCORE)
        for key, rows in masked:
            if j * bc <= key < (j + 1) * bc:
                S16[key - j * bc, rows] = F16(MASKED_SCORE)
        rmax = simd.reduce_lanewise(S16, simd.MAX)
        mnew = simd.reduce_lanewise(np.stack([rmax, m16]), simd.MAX)
        corr16 = exp16(add16(m16, -mnew), a32)
        p16 = exp16(add16(S16, -mnew[None, :]), a32)
        rsum16 = simd.reduce_lanewise(p16, simd.ADD)
        l16 = simd.reduce_lanewise(np.stack([rsum16, mul16(corr16, l16)]), simd.ADD)
        p8 = quant_i8(p16, P8_SCALE).astype(np.int64)                  # [bc, Br]
        pv = V.T @ p8                                                   # [512, Br]
        if j == 0:
            o = pv
        else:
            o = np.clip(np.rint(o.astype(F64) * corr16.astype(F64)[None, :]),
                        -2**31, 2**31 - 1).astype(np.int64) + pv
        m16 = mnew
        tiles.append(dict(S=S, S16=S16, m=m16.copy(), corr=corr16, p16=p16, l=l16.copy(), p8=p8))
    return dict(o=o, m=m16, l=l16, tiles=tiles)


# ---- the layer -------------------------------------------------------------------------------

def shifts(d):
    """The D-port shift of every GEMM of the layer, from its depth (decision 5)."""
    return dict(x=d_shift_for_depth(d.hidden), uk=d_shift_for_depth(d.q_nope),
                uv=d_shift_for_depth(d.kv_rank), s=d_shift_for_depth(d.latent),
                ed=d_shift_for_depth(d.moe_inter), sd=d_shift_for_depth(d.shared_inter))


def kv_row(pack, scales, x16, pos):
    """The cache row [c8 | kpe8] a token with hidden state x16 appends at position pos: the
    device's input norm, W_DKV GEMV, dequantisation, latent norm, RoPE and quantisers, the
    same stages as run() (test_dsv2 pins the two together)."""
    d = pack.W.d
    k = shifts(d)["x"]
    xq = quant_i8(rmsnorm(np.asarray(x16, dtype=F16)), scales.x.inv)
    ckv16 = mul16(gemv(xq, pack.wdkv.q, k)[0], dequant_scale(scales.x.s, pack.wdkv.s, k))
    cos, sin = rope_cos_sin(d, pos)
    kpe = rope(ckv16[d.kv_rank:], cos.astype(F16), sin.astype(F16))
    return np.concatenate([quant_i8(rmsnorm(ckv16[:d.kv_rank]), scales.c.inv),
                           quant_i8(kpe, scales.kpe.inv)])


def mla_token(pack, scales, x16, pos):
    """Stages 1-7 of one token at position pos: everything before the attention."""
    d = pack.W.d
    H = d.heads
    ks = shifts(d)
    x16 = np.asarray(x16, dtype=F16)
    r = dict(x16=x16, pos=pos)
    # ---- 1-4: input norm, quantise, W_Q and W_DKV, latent norm ------------------------------
    xn = rmsnorm(x16)
    xq = quant_i8(xn, scales.x.inv)
    q_raw, _ = gemv(xq, pack.wq.q, ks["x"])
    q_s = dequant_scale(scales.x.s, pack.wq.s, ks["x"])
    q16 = mul16(q_raw, q_s)
    kv_raw, _ = gemv(xq, pack.wdkv.q, ks["x"])
    kv_s = dequant_scale(scales.x.s, pack.wdkv.s, ks["x"])
    ckv16 = mul16(kv_raw, kv_s)
    cn16 = rmsnorm(ckv16[:d.kv_rank])
    r.update(xn=xn, xq=xq, q_raw=q_raw, q_s=q_s, q16=q16, kv_raw=kv_raw, kv_s=kv_s,
             ckv16=ckv16, cn16=cn16)
    # ---- 5: RoPE at position pos, for the 16 q_pe and the one k_pe ---------------------------
    cos, sin = rope_cos_sin(d, pos)
    cos16, sin16 = cos.astype(F16), sin.astype(F16)
    qh = q16.reshape(H, d.q_head)
    q_nope16 = qh[:, :d.q_nope]
    qpe_rot = rope(qh[:, d.q_nope:], cos16, sin16)
    kpe_rot = rope(ckv16[d.kv_rank:], cos16, sin16)
    r.update(cos16=cos16, sin16=sin16, q_nope16=q_nope16, qpe_rot=qpe_rot, kpe_rot=kpe_rot)
    # ---- 6: absorb W_UK, per head -------------------------------------------------------------
    qn8 = quant_i8(q_nope16, scales.qn.inv)
    qt_raw = np.stack([gemv(qn8[h], pack.wuk.q[h], ks["uk"])[0] for h in range(H)])
    qt_s = np.stack([dequant_scale(scales.qn.s, pack.wuk.s[h], ks["uk"]) for h in range(H)])
    qt16 = mul16(qt_raw, qt_s)
    r.update(qn8=qn8, qt_raw=qt_raw, qt_s=qt_s, qt16=qt16)
    # ---- 7: the cache row this token appends ---------------------------------------------------
    r.update(c8_new=quant_i8(cn16, scales.c.inv), kpe8_new=quant_i8(kpe_rot, scales.kpe.inv))
    return r


def mla_output(pack, scales, x16, ot16):
    """Stages 11-13 of one token: its o~ [H, 512] through W_UV and W_O, then the residual."""
    d = pack.W.d
    ks = shifts(d)
    ot8 = quant_i8(ot16, scales.ot.inv)
    oh_raw = np.stack([gemv(ot8[h], pack.wuv.q[h], ks["uv"])[0] for h in range(d.heads)])
    oh_s = np.stack([dequant_scale(scales.ot.s, pack.wuv.s[h], ks["uv"]) for h in range(d.heads)])
    o16 = mul16(oh_raw, oh_s)                                           # [H, 128]
    o8 = quant_i8(o16.reshape(-1), scales.o.inv)
    a_raw, _ = gemv(o8, pack.wo.q, ks["x"])
    a_s = dequant_scale(scales.o.s, pack.wo.s, ks["x"])
    attn16 = mul16(a_raw, a_s)
    return dict(ot16=ot16, ot8=ot8, oh_raw=oh_raw, oh_s=oh_s, o16=o16, o8=o8, a_raw=a_raw,
                a_s=a_s, attn16=attn16, h16=add16(np.asarray(x16, dtype=F16), attn16))


def moe_route(pack, scales, h16):
    """Stages 14-16 of one token: post-attention norm, router, softmax, top-k."""
    d = pack.W.d
    ks = shifts(d)
    hn = rmsnorm(h16)
    hq = quant_i8(hn, scales.h.inv)
    lg_raw, _ = gemv(hq, pack.wr.q, ks["x"])
    lg_s = dequant_scale(scales.h.s, pack.wr.s, ks["x"])
    logits16 = mul16(lg_raw, lg_s)
    p16 = softmax16(logits16)
    ids = top_k16(p16, d.top_k)
    if d.norm_topk or d.routed_scale != 1.0:
        raise NotImplementedError("V2-Lite neither renormalises nor scales the top-k weights")
    return dict(hn=hn, hq=hq, lg_raw=lg_raw, lg_s=lg_s, logits16=logits16, p16=p16, ids=ids,
                w16=p16[ids])


def moe_experts(pack, scales, h16, route, order=None):
    """Stages 17-23 of one token: its slots (in its own top-k order), the shared experts, and
    out = h + shared + sum_i w_i e_i, the slots added in `order` (expert ids; its own top-k
    order by default)."""
    d = pack.W.d
    ks = shifts(d)
    hq = route["hq"]

    def mlp(gu, dn, a_scale, k_gu, k_dn, inter):
        g_raw, _ = gemv(hq, gu.q, k_gu)
        g_s = dequant_scale(scales.h.s, gu.s, k_gu)
        g16 = mul16(g_raw, g_s)
        sg16 = silu16(g16[:inter])
        a16 = mul16(sg16, g16[inter:])
        a8 = quant_i8(a16, a_scale.inv)
        y_raw, _ = gemv(a8, dn.q, k_dn)
        y_s = dequant_scale(a_scale.s, dn.s, k_dn)
        return dict(g_raw=g_raw, g_s=g_s, g16=g16, sg16=sg16, a16=a16, a8=a8, y_raw=y_raw,
                    y_s=y_s, y16=mul16(y_raw, y_s))

    ids = [int(e) for e in route["ids"]]
    slots = []
    for e in ids:
        gu, dn = pack.expert(e)
        slots.append(mlp(gu, dn, scales.expert_a[e], ks["x"], ks["ed"], d.moe_inter))
    shared = mlp(pack.shared_gu, pack.shared_down, scales.shared_a, ks["x"], ks["sd"],
                 d.shared_inter)
    out16 = add16(h16, shared["y16"])
    for e in (ids if order is None else order):
        i = ids.index(e)
        out16 = add16(out16, mul16(route["w16"][i], slots[i]["y16"]))
    return dict(slots=slots, shared=shared, out16=out16)


def union_order(id_lists):
    """The slot order of a pass: the first token's experts in its order, then each later
    token's new ones in its order."""
    order = []
    for ids in id_lists:
        order += [int(e) for e in ids if int(e) not in order]
    return order


def run_tokens(pack, scales, xs16, pos, cache_c8, cache_kpe8, bc=64):
    """Layer 1 for ntok = len(xs16) consecutive tokens in one pass, ntok <= 2 (the Br = 32 query
    lanes hold 16 heads per token): token t sits at position pos + t, every token appends its
    row, and token t attends to keys 0 .. pos + t. The MoE runs the union of the tokens' experts
    (union_order) and each token adds its slots in that order.

    Returns (J, toks): the pass's joint values (the attention) and one dict per token."""
    d = pack.W.d
    H = d.heads
    ntok = len(xs16)
    if ntok not in (1, 2):
        raise ValueError(f"{ntok} tokens: the 32 query lanes hold one or two tokens of {H} heads")
    ks = shifts(d)
    toks = [mla_token(pack, scales, x, pos + t) for t, x in enumerate(xs16)]
    c8 = np.concatenate([cache_c8] + [tk["c8_new"][None] for tk in toks], axis=0)
    kpe8 = np.concatenate([cache_kpe8] + [tk["kpe8_new"][None] for tk in toks], axis=0)
    Kc = np.concatenate([c8, kpe8], axis=1)             # [pos + ntok, 576], the key copy's rows

    # ---- 8-10: query assembly, scores, softmax, weighted sum ---------------------------------
    Br = 2 * H
    Q8 = np.zeros((Br, d.latent), dtype=np.int8)
    for t, tk in enumerate(toks):
        Q8[t * H:(t + 1) * H, :d.kv_rank] = quant_i8(tk["qt16"], scales.qt.inv)
        Q8[t * H:(t + 1) * H, d.kv_rank:] = quant_i8(tk["qpe_rot"], scales.qpe.inv)
    masked = [(pos + u, slice(t * H, (t + 1) * H)) for t in range(ntok) for u in range(t + 1, ntok)]
    a_exp = d.softmax_scale * scales.qt.s * scales.c.s * 2.0 ** ks["s"]
    att = mla_attention(Q8, Kc, c8, a_exp, ks["s"], bc, masked)
    # The last PV leaves through the D port as FP16, O16 = RNE(O * 2^-k_o). Each query's
    # factor c / l, c = 2^k_o s_c / 127, is ONE task: the l beat read twice, Map RSQRT with
    # a = 1/c on both copies (sqrt(c / l), whose square cannot overflow the way l^2 would),
    # then EW1 MUL of the pair. A sticky MUL applies it to O16.
    k_o = d_shift_for_depth(Kc.shape[0])
    O16 = d_port(att["o"], k_o)                                       # [512, Br], O^T
    a_n = F32(P8_SCALE / (2.0 ** k_o * scales.c.s))
    t16 = simd.stream_map(att["l"], a_n, 0.0, simd.RSQRT)
    rsc16 = mul16(t16, t16)
    ot_all = mul16(O16, rsc16[None, :]).T                             # [Br, 512]
    J = dict(ks=ks, Kc=Kc, Vc=c8, Q8=Q8, a_exp=float(F32(a_exp)), att=att, k_o=k_o, O16=O16,
             a_n=float(a_n), rsc16=rsc16, masked=masked)

    # ---- 11-23: per token: W_UV, W_O, residual, then the MoE -----------------------------------
    for t, tk in enumerate(toks):
        tk.update(mla_output(pack, scales, tk["x16"], ot_all[t * H:(t + 1) * H]))
        tk.update(moe_route(pack, scales, tk["h16"]))
    order = union_order([tk["ids"] for tk in toks])
    J["order"] = order
    for tk in toks:
        tk.update(moe_experts(pack, scales, tk["h16"], tk, [e for e in order if e in list(tk["ids"])]))
    return J, toks


def run(pack, scales, x16, pos, cache_c8, cache_kpe8, bc=64):
    """Layer 1 for one token on the device (run_tokens with one token, as one dict). cache_* are
    the L cached rows (INT8)."""
    J, (tk,) = run_tokens(pack, scales, [x16], pos, cache_c8, cache_kpe8, bc)
    r = dict(tk)
    r.update(J)
    return r
