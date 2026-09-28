# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""One golden pack per (seed, L): layer 1 for one new token at position L, L tokens cached.

    python3 -m dsv2.golden --seed 1 --L 511          (from target/snitch_cluster/sw/apps/dsv2)

prints, stage by stage, how far the device model (hwmodel.py) is from the float reference
(reference.py). Kernel data generators call make() and take the tensors their stage needs.

WHAT A PACK HOLDS. Random BF16 weights at the real shapes (weights.py) and their packed
INT8 form (pack.py); L previous tokens whose cache rows are built by the layer's own kv
path (reference.cache_entry) and quantised with the calibrated static scales; the scales
(calib.py, from sample tokens that are NOT the golden token); the golden token x; and
both models' every intermediate.

THE GOLDEN TOKEN is the first draw whose routing is unambiguous on the device: the
device model's top 6 must equal the reference's, and its sixth and seventh FP16 routing
weights must be at least TIE_ULP apart, so the device's own softmax (within a few ULP of
the model) cannot swap them.
"""

import argparse
from dataclasses import dataclass

import numpy as np

from . import calib, hwmodel, model, reference
from .fp import F16, F32, quant_i8, ulp16
from .pack import LayerPack
from .weights import LayerWeights

TIE_ULP = 8


@dataclass
class Golden:
    seed: int
    L: int
    pos: int
    bc: int
    dims: model.Dims
    W: LayerWeights
    pack: LayerPack
    cache: reference.Cache      # float, the reference's
    c8: np.ndarray              # [L, 512] the cached latents as the device holds them
    kpe8: np.ndarray            # [L, 64]
    scales: hwmodel.Scales
    x16: np.ndarray             # the golden token's hidden state
    ref: dict
    hw: dict
    tries: int


def _draw_states(rng, n, hidden):
    return rng.normal(0.0, 1.0, size=(n, hidden)).astype(F16)


def make(seed=1, L=511, n_calib=8, bc=64, max_tries=32):
    d = model.load()
    W = LayerWeights(d, seed)
    P = LayerPack(W)
    rng = np.random.default_rng([seed, 2])
    prev = _draw_states(rng, L, d.hidden).astype(F32)
    cache = reference.build_cache(W, prev, range(L))
    xs_cal = _draw_states(rng, n_calib, d.hidden).astype(F32)
    for tries in range(1, max_tries + 1):
        x16 = _draw_states(rng, 1, d.hidden)[0]
        R = reference.layer(W, x16.astype(F32), L, cache)
        S = calib.calibrate(W, cache, L, xs_cal, experts=R["moe_ids"])
        c8 = quant_i8(cache.c_unit.astype(F16), S.c.inv)
        kpe8 = quant_i8(cache.kpe.astype(F16), S.kpe.inv)
        H = hwmodel.run(P, S, x16, L, c8, kpe8, bc)
        order = np.argsort(-H["p16"].astype(np.float64), kind="stable")
        gap = int(ulp16(H["p16"][order[d.top_k - 1]], H["p16"][order[d.top_k]]))
        if list(H["ids"]) == list(R["moe_ids"]) and gap >= TIE_ULP:
            return Golden(seed, L, L, bc, d, W, P, cache, c8, kpe8, S, x16, R, H, tries)
    raise RuntimeError(f"no token in {max_tries} draws routes unambiguously")


def spec_passes(g, starts, ntok=1):
    """Passes after g's cache: pass p runs ntok tokens at positions starts[p], starts[p] + 1,
    ... -- the first token of pass 0 is g's own, every other a fresh draw. A pass reads the cache
    rows below its start and appends its tokens' rows: consecutive starts continue the sequence,
    and a start below the end of the pass before drops that pass's last rows (a rejected draft),
    which the new rows overwrite.

    Returns one dict per pass: pos, xs (the tokens), J and toks (hwmodel.run_tokens), and the
    cache rows after it (c8, kpe8)."""
    rng = np.random.default_rng([g.seed, 17])
    c8, kpe8 = g.c8, g.kpe8
    out = []
    for p, start in enumerate(starts):
        if start > c8.shape[0]:
            raise ValueError(f"pass {p} starts at {start}, past the {c8.shape[0]} rows cached")
        xs = [g.x16 if p == 0 and t == 0 else _draw_states(rng, 1, g.dims.hidden)[0]
              for t in range(ntok)]
        J, toks = hwmodel.run_tokens(g.pack, g.scales, xs, start, c8[:start], kpe8[:start], g.bc)
        c8 = np.concatenate([c8[:start]] + [tk["c8_new"][None] for tk in toks])
        kpe8 = np.concatenate([kpe8[:start]] + [tk["kpe8_new"][None] for tk in toks])
        out.append(dict(pos=start, xs=xs, J=J, toks=toks, c8=c8, kpe8=kpe8))
    return out


# ---- the per-stage report ---------------------------------------------------------------------

def _cmp(hw, rf):
    a = np.asarray(hw, dtype=np.float64).reshape(-1)
    b = np.asarray(rf, dtype=np.float64).reshape(-1)
    rel = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
    cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
    return rel, float(np.max(np.abs(a - b))), cos


def stages(g):
    """(name, device value, reference value in the device's domain) for every stage."""
    d, H, R = g.dims, g.hw, g.ref
    gc = g.W.kv_norm.astype(np.float64)
    s = g.scales
    qh = R["att_q"].reshape(d.heads, d.q_head)
    att = H["att"]
    T = g.L + 1
    # the device's scores in logit units: S * s_q~ * s_c * softmax_scale
    s_dev = np.concatenate([t["S"] for t in att["tiles"]], axis=0)[:T, :d.heads].T
    s_dev = s_dev * s.qt.s * s.c.s * d.softmax_scale
    p_dev = np.concatenate([t["p16"] for t in att["tiles"]], axis=0)[:T, :d.heads].T
    yield "xn (unit)", H["xn"], reference.rmsnorm_unit(R["x"], d.rms_eps)
    yield "q = xn W_Q", H["q16"], R["att_q"]
    yield "ckv = xn W_DKV", H["ckv16"], R["att_ckv"]
    yield "c (unit latent)", H["cn16"], R["att_c_new"]
    yield "q_pe rotated", H["qpe_rot"], R["att_q_pe"]
    yield "k_pe rotated", H["kpe_rot"], R["att_kpe_new"]
    yield "q_nope", H["q_nope16"], qh[:, :d.q_nope]
    yield "q~ = q_nope W_UK g", H["qt16"], R["att_q_lat"] * gc[None, :]
    yield f"scores [{d.heads} x {T}]", s_dev, R["att_s_lat"]
    yield "softmax max m", att["m"][:d.heads].astype(np.float64) * s.qt.s * s.c.s * 2.0 ** H["ks"]["s"] \
        * d.softmax_scale, R["att_s_lat"].max(axis=1)
    yield "p (last tile, unnormalised)", p_dev[:, -g.bc:], np.exp(
        R["att_s_lat"] - R["att_s_lat"].max(axis=1, keepdims=True))[:, -g.bc:]
    yield "o~ (unit latent)", H["ot16"], R["att_o_lat"] / gc[None, :]
    yield "o = o~ W_UV g", H["o16"], R["att_o"]
    yield "attn = o W_O", H["attn16"], R["att_attn_out"]
    yield "h = x + attn", H["h16"], R["h"]
    yield "hn (unit)", H["hn"], reference.rmsnorm_unit(R["h"], d.rms_eps)
    yield "router logits", H["logits16"], R["moe_logits"]
    yield "router probs", H["p16"], R["moe_probs"]
    for i, (e, sl) in enumerate(zip(H["ids"], H["slots"])):
        yield f"slot {i} (expert {e:2d}) act", sl["a16"], R["moe_expert_acts"][i]
        yield f"slot {i} (expert {e:2d}) out", sl["y16"], R["moe_experts"][i]
    yield "shared act", H["shared"]["a16"], R["moe_shared_act"]
    yield "shared out", H["shared"]["y16"], R["moe_shared"]
    yield "out = h + moe", H["out16"], R["out"]


def report(g):
    d, H = g.dims, g.hw
    lines = [f"DeepSeek-V2-Lite layer 1, seed {g.seed}: one token at position {g.pos}, "
             f"{g.L} cached, Bc = {g.bc} ({-(-(g.L + 1) // g.bc)} key tiles); golden token draw "
             f"{g.tries}",
             f"top-{d.top_k}: device {list(map(int, H['ids']))}  reference "
             f"{list(map(int, g.ref['moe_ids']))}",
             f"D-port shifts: {H['ks']}  O: {H['k_o']}   exp scale a' = {H['a_exp']:.6g}",
             f"{'stage':34s} {'rel L2 err':>11s} {'max |err|':>11s} {'cosine':>9s}"]
    for name, a, b in stages(g):
        rel, mx, cos = _cmp(a, b)
        lines.append(f"{name:34s} {rel:11.3e} {mx:11.3e} {cos:9.6f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--L", type=int, default=511)
    ap.add_argument("--bc", type=int, default=64)
    ap.add_argument("--calib", type=int, default=8)
    a = ap.parse_args()
    print(report(make(a.seed, a.L, a.calib, a.bc)))


if __name__ == "__main__":
    main()
