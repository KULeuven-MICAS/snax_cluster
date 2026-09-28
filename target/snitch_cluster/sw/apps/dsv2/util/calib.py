# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Static activation scales, calibrated offline from sample tokens.

The SIMD quantiser takes one FP32 scale per task and the SIMD has no abs-max, so every
activation the device quantises gets ONE static scale per tensor: amax over the sample
tokens, mapped onto 127. The samples run through the float reference, and each tensor is
taken in the domain the DEVICE quantises it in -- which differs from the reference's
where a norm gain is folded into the weights:

    xn, hn      the RMS-normalised states WITHOUT the gain (it is in W_Q, W_DKV, the router
                and the experts' gate and up)
    c           the latent without the kv_a_layernorm gain; the cache rows count too, since
                they are quantised with the same scale
    q~          q_lat * g: the gain moved from c onto the query side (W_UK absorbs it)
    o~          o_lat / g: the weighted sum of UNIT latents (W_UV absorbs the gain)

s_q~ and s_qpe are tied, s_q~ * s_c = s_qpe * s_kpe, so every score is one exact INT32
dot product over [c | k_pe]: the common product is set by whichever segment needs more.
"""

import numpy as np

from . import reference as ref
from .hwmodel import QScale, Scales

F32 = np.float32


def amax(*arrays):
    return float(max(np.max(np.abs(np.asarray(a, dtype=np.float64))) for a in arrays))


class ExpertScales(dict):
    """Routed expert id -> the static scale of its SwiGLU output. Every sample goes through the
    expert, whether or not the router would pick it for that sample. An expert no token has used
    yet is calibrated when first asked for: routing does not depend on these scales."""

    def __init__(self, W, hns, margin):
        super().__init__()
        self._W, self._hns, self._margin = W, hns, margin

    def __missing__(self, e):
        acts = [ref.swiglu_mlp(hn, self._W.expert(int(e)))[1] for hn in self._hns]
        self[e] = QScale.for_amax(self._margin * amax(*acts))
        return self[e]


def calibrate(W, cache, pos, xs, experts=(), margin=1.0):
    """Scales from sample hidden states xs [n, 2048] at position pos against `cache`.

    `experts` are calibrated now; any other routed expert when a token first routes to it.
    """
    d = W.d
    g_c = W.kv_norm.astype(np.float64)
    acc = {k: [] for k in ("xn", "qn", "c", "kpe", "qt", "qpe", "ot", "o", "hn", "sa")}
    hns = []
    for x in xs:
        R = ref.layer(W, x, pos, cache)
        acc["xn"].append(ref.rmsnorm_unit(R["x"], d.rms_eps))
        acc["qn"].append(R["att_q_nope"])
        acc["c"].append(R["att_c_new"])
        acc["kpe"].append(R["att_kpe_new"])
        acc["qt"].append(R["att_q_lat"] * g_c[None, :])
        acc["qpe"].append(R["att_q_pe"])
        acc["ot"].append(R["att_o_lat"] / g_c[None, :])
        acc["o"].append(R["att_o"])
        hn_unit = ref.rmsnorm_unit(R["h"], d.rms_eps)
        acc["hn"].append(hn_unit)
        acc["sa"].append(R["moe_shared_act"])
        hns.append(R["hn"])

    def q(key, *extra):
        return QScale.for_amax(margin * amax(*acc[key], *extra))

    s_c = q("c", cache.c_unit)
    s_kpe = q("kpe", cache.kpe)
    prod = max(amax(*acc["qt"]) * s_c.s, amax(*acc["qpe"]) * s_kpe.s) * margin / 127.0
    s_qt = QScale(float(F32(s_c.s / prod)))
    s_qpe = QScale(float(F32(s_kpe.s / prod)))
    expert_a = ExpertScales(W, hns, margin)
    for e in experts:
        expert_a[int(e)]
    return Scales(x=q("xn"), qn=q("qn"), c=s_c, kpe=s_kpe, qt=s_qt, qpe=s_qpe, ot=q("ot"),
                  o=q("o"), h=q("hn"), shared_a=q("sa"), expert_a=expert_a)
