# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""What the block kernels (sw/apps/dsv2/include/snax-dsv2-mla.h, snax-dsv2-moe.h) read from data.h,
emitted once for every app that runs them: the model's constants, the weights, the cache and
the passes. The apps add their own goldens.

    model_defines   dimensions, D-port shifts, static scales             both blocks
    mla_weights     wdkv, wq, wuk, wuv, wo and their factors s_kv .. s_o   the MLA block
    cache           the two DRAM copies holding the L cached rows          the MLA block
    mla_passes      per pass: x16_<p>, rope_cos_<p>, rope_sin_<p>, NTOK and the
                    MLA_PASSES initialiser of dsv2_mla_pass_t[NP]           the MLA block
    moe_weights     the router, the expert table (the experts named), the
                    shared experts                                          the MoE block
"""

import numpy as np

from .fp import F16, bits16, f32bits
from .hwmodel import P8_SCALE, dequant_scale
from .layout import key_copy, value_copy


def model_defines(em, g, bc, cap):
    d, S, H = g.dims, g.scales, g.hw
    ks = H["ks"]
    em.define("HID", d.hidden)
    em.define("HEADS", d.heads)
    em.define("Q_HEAD", d.q_head, "a head's slice of q: [q_nope | q_pe]")
    em.define("Q_NOPE", d.q_nope)
    em.define("ROPE_DIM", d.q_rope)
    em.define("KV_RANK", d.kv_rank)
    em.define("V_HEAD", d.v_head)
    em.define("N_EXP", d.n_routed)
    em.define("TOP_K", d.top_k)
    em.define("I_EXP", d.moe_inter)
    em.define("I_SH", d.shared_inter)
    em.define("BC", bc, "keys per attention tile")
    em.define("CAP", cap, "tokens the cache copies hold")
    em.define("K_X", ks["x"], "D-port shift at depth 2048 (W_DKV, W_Q, W_O, router, gate|up)")
    em.define("K_UK", ks["uk"], "at depth 128 (W_UK)")
    em.define("K_S", ks["s"], "the scores, depth 576")
    em.define("K_UV", ks["uv"], "at depth 512 (W_UV)")
    em.define("K_ED", ks["ed"], "at depth 1408 (an expert's down)")
    em.define("K_SD", ks["sd"], "at depth 2816 (the shared down)")
    for name, sc, what in (("INV_X", S.x, "the normed input"), ("INV_C", S.c, "the normed latent"),
                           ("INV_KPE", S.kpe, "the rotated k_pe"), ("INV_QN", S.qn, "q_nope"),
                           ("INV_QT", S.qt, "q~"), ("INV_QPE", S.qpe, "the rotated q_pe"),
                           ("INV_OT", S.ot, "o~"), ("INV_O", S.o, "the heads' outputs"),
                           ("INV_H", S.h, "the normed post-attention state")):
        em.define(name, f"0x{f32bits(sc.inv):08X}u", f"quantise {what}")
    em.define("A_EXP", f"0x{f32bits(H['a_exp']):08X}u", f"exp scale a' = {H['a_exp']:.6g}")
    em.define("P8_INV", f"0x{f32bits(P8_SCALE):08X}u", "P8 = rne(127 P)")


def mla_weights(em, g, mesh):
    P, H = g.pack, g.hw
    em.blob("wdkv", P.wdkv.blob(mesh))
    em.blob("wq", P.wq.blob(mesh))
    em.blob("wuk", P.wuk.blob(mesh))
    em.blob("wuv", P.wuv.blob(mesh))
    em.blob("wo", P.wo.blob(mesh))
    em.blob("s_kv", bits16(H["kv_s"]))
    em.blob("s_q", bits16(H["q_s"]))
    em.blob("s_uk", bits16(H["qt_s"]).reshape(-1))
    em.blob("s_uv", bits16(H["oh_s"]).reshape(-1))
    em.blob("s_o", bits16(H["a_s"]))


def cache_rows(g):
    """The L cached rows [c8 | kpe8], [L, 576] int8."""
    return np.concatenate([g.c8, g.kpe8], axis=1)


def cache(em, g, cap, mesh):
    """Both copies holding the L cached rows; written, since the kernel appends to them."""
    rows0 = cache_rows(g)
    em.blob("key", key_copy(rows0, cap, mesh), writable=True)
    em.blob("val", value_copy(rows0[:, :g.dims.kv_rank], cap, mesh), writable=True)


def rope_tables(H, heads):
    """cos repeated per pair and sin carrying the pair's sign, tiled for 16 q_pe and one k_pe."""
    c_rep = np.repeat(H["cos16"], 2)
    s_sgn = np.empty(c_rep.shape[0], dtype=F16)
    s_sgn[0::2] = -H["sin16"]
    s_sgn[1::2] = H["sin16"]
    return np.tile(c_rep, heads + 1), np.tile(s_sgn, heads + 1)


def mla_passes(em, passes, heads):
    """passes: golden.spec_passes' dicts (pos, xs, toks, J). Emits each pass's inputs -- its
    tokens [NTOK, HID] and their RoPE tables, a row per token -- and MLA_PASSES, the
    initialiser of `dsv2_mla_pass_t[NP]` (snax-dsv2-mla.h: x16, rope_cos, rope_sin, pos, k_o,
    a_n)."""
    rows = []
    for p, q in enumerate(passes):
        tabs = [rope_tables(tk, heads) for tk in q["toks"]]
        em.blob(f"x16_{p}", bits16(np.concatenate([np.asarray(x, dtype=F16) for x in q["xs"]])))
        em.blob(f"rope_cos_{p}", bits16(np.concatenate([c for c, _ in tabs])))
        em.blob(f"rope_sin_{p}", bits16(np.concatenate([s for _, s in tabs])))
        rows.append(f"{{x16_{p}, rope_cos_{p}, rope_sin_{p}, {q['pos']}u, {q['J']['k_o']}u, "
                    f"0x{f32bits(q['J']['a_n']):08X}u}}")
    em.define("NP", len(passes), "passes")
    em.define("NTOK", len(passes[0]["xs"]), "tokens per pass")
    em.define("MLA_PASSES", "{" + ", ".join(rows) + "}")


def golden_pass(g):
    """The golden pack's own token as one pass, in golden.spec_passes' form."""
    return dict(pos=g.pos, xs=[g.x16], toks=[g.hw], J=g.hw)


def moe_weights(em, g, ids, mesh):
    """The router, the 64-entry expert table with weights for `ids` only (every other entry is
    zero and refused by the kernel), and the shared experts."""
    P, H, S = g.pack, g.hw, g.scales
    ks = H["ks"]
    em.blob("wr", P.wr.blob(mesh))
    em.blob("wr_s", bits16(H["lg_s"]))

    def expert_blobs(tag, gu, dn, a_scale, k_dn):
        em.blob(f"{tag}_gu", gu.blob(mesh))
        em.blob(f"{tag}_gu_s", bits16(dequant_scale(S.h.s, gu.s, ks["x"])))
        em.blob(f"{tag}_dn", dn.blob(mesh))
        em.blob(f"{tag}_dn_s", bits16(dequant_scale(a_scale.s, dn.s, k_dn)))
        return f"{{{tag}_gu, {tag}_gu_s, {tag}_dn, {tag}_dn_s, 0x{f32bits(a_scale.inv):08X}u}}"

    rows = ["{0, 0, 0, 0, 0}"] * g.dims.n_routed
    for e in sorted(set(int(i) for i in ids)):
        gu, dn = P.expert(e)
        rows[e] = expert_blobs(f"e{e}", gu, dn, S.expert_a[e], ks["ed"])
    shared = expert_blobs("sh", P.shared_gu, P.shared_down, S.shared_a, ks["sd"])
    em.c("typedef struct {\n    const int8_t *gu;\n    const uint16_t *gu_s;\n    const int8_t *dn;\n"
         "    const uint16_t *dn_s;\n    uint32_t inv_a;  // the SwiGLU output's quantiser, FP32\n"
         "} expert_t;")
    em.c("static const expert_t experts[N_EXP] = {\n" +
         ",\n".join(f"    {r}" for r in rows) + "\n};")
    em.c(f"static const expert_t shared_expert = {shared};")
