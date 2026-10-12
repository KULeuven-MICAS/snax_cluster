# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Checks of the golden generator itself. Run from target/snitch_cluster/sw/apps/dsv2:

    python3 -m util.test_dsv2          (or: pytest util/test_dsv2.py)

Each test is small and fast; the whole-layer check (hwmodel against the reference) is
golden.py's report.
"""

import os
import sys

import numpy as np

from . import model, reference
from .fp import F16, F32, bits16, d_port, d_shift_for_depth, quant_i8
from .layout import (MESH, b_chunks, from_a, from_b, gemv_a, pack_int4, to_a, to_b, to_a4,
                     to_b_pairs, unpack_int4)
from .pack import LayerPack, pack, unpack
from .weights import LayerWeights

# snax_utils, the RTL's golden models, is in the repository's util/sim
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             *[".."] * 6, "util", "sim"))
from snax_utils import block_gemm_golden_model, int32_to_fp16_golden  # noqa E402

_CACHE = {}


def _layer():
    if "W" not in _CACHE:
        W = LayerWeights(model.load(), seed=3)
        _CACHE["W"], _CACHE["P"] = W, LayerPack(W)
    return _CACHE["W"], _CACHE["P"]


def test_d_port_matches_rtl_scalar_model():
    rng = np.random.default_rng(0)
    acc = np.concatenate([
        rng.integers(-2**31, 2**31, size=4000),
        rng.integers(-70000, 70000, size=4000),
        np.array([0, 1, -1, 2047, 2048, 2049, 65504, 65519, 65520, 65536, 2**31 - 1, -2**31]),
    ]).astype(np.int64)
    for k in (0, 5, 7, 8, 9, 10, 14, 15):
        got = bits16(d_port(acc, k))
        want = np.array([int32_to_fp16_golden(int(v), k) for v in acc], dtype=np.uint16)
        assert np.array_equal(got, want), f"d_port differs from the RTL model at k = {k}"


def test_d_shift_for_depth_matches_the_plan():
    # decision 5: 127^2 * K <= 65,504 * 2^k
    want = {128: 5, 512: 7, 576: 8, 1408: 9, 2048: 9, 2816: 10}
    assert {K: d_shift_for_depth(K) for K in want} == want


def test_quant_i8_rounds_to_even_and_saturates_symmetrically():
    x = np.array([0.5, 1.5, 2.5, -0.5, -1.5, 126.5, 127.4, 200.0, -200.0, -127.6], dtype=F16)
    assert quant_i8(x, 1.0).tolist() == [0, 2, 2, 0, -2, 126, 127, 127, -127, -127]


def test_layouts_round_trip():
    rng = np.random.default_rng(1)
    R = rng.integers(-128, 128, size=(32, 64)).astype(np.int8)
    assert np.array_equal(from_a(to_a(R), 32, 64), R)
    assert np.array_equal(from_b(to_b(R), 32, 64), R)


def test_b_chunks_are_slices_of_the_whole_b_layout():
    rng = np.random.default_rng(2)
    W = rng.integers(-127, 128, size=(256, 192)).astype(np.int8)
    assert np.array_equal(b_chunks(W, 64), to_b(W))


def test_gemv_over_the_block_model():
    """The A/B layouts drive util/sim/snax_utils.block_gemm_golden_model to x @ W in row 0."""
    rng = np.random.default_rng(4)
    K, N = 128, 64
    x = rng.integers(-127, 128, size=K).astype(np.int8)
    W = rng.integers(-127, 128, size=(K, N)).astype(np.int8)
    mr, ts, mc = MESH
    D = block_gemm_golden_model(1, K // ts, N // mc, mr, ts, mc, gemv_a(x), to_b(W), 0, 0,
                                np.zeros(mr * N, dtype=np.int64))
    D = np.asarray(D).reshape(N // mc, mr, mc)            # [n][row][col]
    assert np.array_equal(D[:, 0, :].reshape(-1), x.astype(np.int64) @ W.astype(np.int64))
    assert not D[:, 1:, :].any(), "rows 1..15 of the padded A must give zeros"


def test_pack_round_trip_and_chunked_gemv():
    """F3: unpack(pack(W)) == W, and a GEMV over the streamed chunks equals the model's."""
    W, P = _layer()
    for p in (P.wq, P.wdkv, P.wo, P.wr, P.shared_gu, P.shared_down):
        assert np.array_equal(unpack(p.blob(), p.K, p.N), p.q), p.name
    rng = np.random.default_rng(5)
    for p in (P.wq, P.wr, P.shared_down):
        x = rng.integers(-127, 128, size=p.K).astype(np.int64)
        blob = p.blob()
        y = []
        for j in range(p.N // 64):
            chunk = blob[j * p.K * 64:(j + 1) * p.K * 64]
            y.append(x @ from_b(chunk, p.K, 64).astype(np.int64))
        assert np.array_equal(np.concatenate(y), x @ p.q.astype(np.int64)), p.name


def test_packed_weights_approximate_the_folded_weights():
    # Half a step, plus the float32 rounding of W'/s and q*s (a few 1e-6 of a step).
    W, P = _layer()
    half = 0.5 * (1.0 + 1e-5)
    want = W.wq * W.in_norm[:, None]
    err = np.abs(P.wq.dequant() - want).max(axis=0)
    assert np.all(err <= half * P.wq.s), "per-column INT8 error exceeds half a step"
    # the absorbed W_UK carries the latent gain on its columns
    h = 5
    want = W.w_uk(h).T * W.kv_norm[None, :]
    assert np.all(np.abs(P.wuk.dequant()[h] - want).max(axis=0) <= half * P.wuk.s[h])


# ---- INT4 weights through B's converter -----------------------------------------------------

def _b_beats(blob, loops, mask, s1=0):
    """snax_split_cluster's B reader: 16 channels of 8 bytes, channel i at 8 (i % 8) + s1 (i // 8),
    a disabled channel presenting zero, walking `loops` [(bound, stride), ...] innermost first.
    Returns the beats, [passes, 128] uint8, channel 0 in bytes 0..7."""
    mem = np.asarray(blob).view(np.uint8)
    beats = []
    for idx in np.ndindex(*[b for b, _ in reversed(loops)]):
        off = sum(i * st for i, (_, st) in zip(idx, reversed(loops)))
        beat = np.zeros(128, dtype=np.uint8)
        for ch in range(16):
            if mask >> ch & 1:
                a = off + 8 * (ch % 8) + s1 * (ch // 8)
                beat[8 * ch:8 * ch + 8] = mem[a:a + 8]
        beats.append(beat)
    return np.stack(beats)


def _int4_converter(beats):
    """B's IntlowToInthighConverter: nibble i of the beat's low half, sign-extended into byte i."""
    return np.stack([unpack_int4(b[:64]).view(np.uint8) for b in beats])


def test_int4_packs_and_unpacks():
    v = np.array([-8, 7, -1, 0, 3, -5, 7, -8], dtype=np.int8)
    assert pack_int4(v).view(np.uint8).tolist() == [0x78, 0x0F, 0xB3, 0x87]
    assert np.array_equal(unpack_int4(pack_int4(v)), v)


def test_int4_b_walks_feed_the_int8_beats():
    """snax-dsv2.h's INT4 B walks over the paired, nibble-packed blob, through the converter,
    hand the array the beats its INT8 walks read from the INT8 blob of the same values:
    dsv2_gemv1_arm (two blocks a pass, groups) and dsv2_gemm_arm (one block a pass, row tiles).
    dsv2_gemm_arm's INT4 walk advances 64 B a pass within a block, as its INT8 walk does."""
    rng = np.random.default_rng(7)
    K, nb, groups = 64, 4, 2
    kt, N = K // 4, 16 * nb
    q = [rng.integers(-8, 8, size=(K, N)).astype(np.int8) for _ in range(groups)]
    b8 = np.concatenate([to_b(m) for m in q])
    b4 = np.concatenate([pack_int4(to_b_pairs(m)) for m in q])
    want = _b_beats(b8, [(kt, 64), (nb // 2, 2 * kt * 64), (groups, kt * 64 * nb)], 0xFFFF,
                    s1=kt * 64)
    got = _int4_converter(_b_beats(b4, [(kt, 64), (nb // 2, kt * 64), (groups, kt * 32 * nb)],
                                   0xFF))
    assert np.array_equal(got, want), "(1, 4, 32)"
    mt = 3
    want = _b_beats(b8[:K * N], [(kt, 64), (nb, kt * 64), (mt, 0)], 0xFF)
    got = _int4_converter(_b_beats(b4[:K * N // 2],
                                   [(kt, 64), (2, 32), (nb // 2, kt * 64), (mt, 0)], 0x0F))
    assert np.array_equal(got, want), "(16, 4, 16)"


def test_compact_a_walks_read_the_tokens_rows():
    """The compact A operand of several tokens (snax-dsv2.h, SEVERAL TOKENS) two ways: to_a4's
    bytes, and the A reader's walk (T0 8 x 64 B, T1 4 x 16 B, T2 kt / 32 x 512 B, channels 0 and
    1), which hands the array pass k the tokens' four values at 4 k in rows 0..3. Within each run
    of eight blocks consecutive passes read 64 B apart, the stride B advances at. (The SIMD's
    writes of the same bytes: test_a_row_writes_rebuild_the_gemv_operands.)"""
    rng = np.random.default_rng(11)
    K, T = 384, 4
    kt = K // 4
    xs = rng.integers(-128, 128, size=(T, K)).astype(np.int8)
    a4 = to_a4(xs)
    assert a4.size == 4 * K
    addrs = [b0 + b1 + b2 for b2 in range(0, 512 * (kt // 32), 512) for b1 in range(0, 64, 16)
             for b0 in range(0, 512, 64)]
    assert all(addrs[k + 1] - addrs[k] == 64 for k in range(kt - 1) if k % 8 != 7)
    for k, a in enumerate(addrs):
        rows = a4[a:a + 16].reshape(4, 4)
        assert np.array_equal(rows, xs[:, 4 * k:4 * k + 4]), f"pass {k}"


# ---- the SIMD core's AGUs (snax-simd-lib.h shapes), at the cfg's beat ------------------------

def _walk(loops):
    """The offsets a shape's loops [(bound, stride), ...] (innermost first) visit, in order."""
    return [sum(i * st for i, (_, st) in zip(idx, reversed(loops)))
            for idx in np.ndindex(*[b for b, _ in reversed(loops)])]


def _simd_read(mem, base, lane_stride, mask, loops, chan):
    """The SIMD reader: per beat, channel c reads 8 bytes at the beat's address + lane_stride c,
    a disabled channel presenting zero. Returns [beats, 8 chan] uint8."""
    beats = []
    for off in _walk(loops):
        beat = np.zeros(8 * chan, dtype=np.uint8)
        for c in range(chan):
            if mask >> c & 1:
                a = base + off + lane_stride * c
                beat[8 * c:8 * c + 8] = mem[a:a + 8]
        beats.append(beat)
    return np.stack(beats)


def _simd_write(mem, base, lane_stride, byte_mask, loops, chan, beats):
    """The SIMD writer: channel c of each beat to the beat's address + lane_stride c, the bytes
    byte_mask selects of every 8-byte word."""
    offs = _walk(loops)
    assert len(offs) == len(beats), f"{len(beats)} beats for a walk of {len(offs)}"
    for off, beat in zip(offs, beats):
        for c in range(chan):
            for b in range(8):
                if byte_mask >> b & 1:
                    mem[base + off + lane_stride * c + b] = beat[8 * c + b]


def _a_row_write(x16, ops, a, n, row, compact, chan):
    """snax-dsv2.h dsv2_a_row_shapes and the chain it feeds, on a byte image: x16 holds the n
    FP16 values (and, ops = 2, a second operand of ones at +od), written into row `row` of the A
    operand `a` (uint8, modified). The quantiser runs at inv_scale 1 on integer values."""
    lanes = 4 * chan
    od = 2 * n + 64
    mem = np.zeros(od * ops + 64, dtype=np.uint8)
    pad = 64  # x sits 64 bytes in, so an odd row's read 4 bytes lower stays inside the image
    mem[pad:pad + 2 * n] = np.asarray(x16, dtype=F16).view(np.uint8)
    if ops > 1:
        mem[pad + od:pad + od + 2 * n] = np.ones(n, dtype=F16).view(np.uint8)
    odd = row & 1
    loops = ([(ops, od)] if ops > 1 else []) + [(2, lanes), (n // lanes, 2 * lanes)]
    beats = _simd_read(mem, pad - 4 * odd, 4, 0x55555555 << odd, loops, chan)
    lanes16 = beats.view(F16).astype(np.float32)
    if ops > 1:  # StreamElementwise MUL of each operand pair
        lanes16 = lanes16[0::2] * lanes16[1::2]
    q = quant_i8(lanes16.astype(F16), 1.0).reshape(-1, 2 * lanes).view(np.uint8)  # 2 beats -> 1
    if compact:
        oloops = [(8 // chan, 64 * chan), (4, 16), (n // 128, 512)]
    else:
        oloops = [(n // lanes, 64 * chan)]
    _simd_write(a, 8 * (row // 2), 64, 0x0F << (4 * odd), oloops, chan, q)


def test_a_row_writes_rebuild_the_gemv_operands():
    """The quantiser's row writes (snax-dsv2.h dsv2_a_row_shapes, at the cfg's SIMD beat) put x
    in row 0 of the 16-row layout (gemv_a) and token t in row t of the compact one (to_a4),
    writing nothing else, plain or fused with a MUL, even rows and odd."""
    from . import simd
    chan = simd.LANES // 4
    rng = np.random.default_rng(12)
    n = 384
    xs = rng.integers(-127, 128, size=(4, n)).astype(np.int8)
    for ops in (1, 2):
        a = np.zeros(16 * n, dtype=np.uint8)
        _a_row_write(xs[0].astype(F16), ops, a, n, 0, False, chan)
        assert np.array_equal(a.view(np.int8), gemv_a(xs[0])), f"16-row layout, ops {ops}"
        a = np.zeros(4 * n, dtype=np.uint8)
        for t in range(4):
            _a_row_write(xs[t].astype(F16), ops, a, n, t, True, chan)
        assert np.array_equal(a.view(np.int8), to_a4(xs)), f"compact layout, ops {ops}"
    # an odd row of the 16-row layout: row 1 alone
    a = np.zeros(16 * n, dtype=np.uint8)
    _a_row_write(xs[1].astype(F16), 1, a, n, 1, False, chan)
    R = np.zeros((16, n), dtype=np.int8)
    R[1] = xs[1]
    assert np.array_equal(a.view(np.int8), to_a(R)), "16-row layout, row 1"


def test_reduce_follows_the_rtls_order():
    """simd.reduce_lanes against the order SimdStreamReduce computes, written out: beat k into
    partial k mod P, the partials folded by a balanced tree, unwritten ones the identity."""
    from . import simd
    rng = np.random.default_rng(13)
    for nb in (1, 2, 3, 4, 7, 11):
        x = (rng.standard_normal((nb, 4)) * 300).astype(F16)
        xf = x.astype(F32)
        for P in (1, 3, 5):
            parts = [np.zeros(4, dtype=F32) for _ in range(P)]
            for k in range(nb):
                parts[k % P] = (parts[k % P] + xf[k]).astype(F32)
            lvl = parts
            while len(lvl) > 1:
                lvl = [(lvl[i] + lvl[i + 1]).astype(F32) for i in range(0, len(lvl) - 1, 2)] + \
                    ([lvl[-1]] if len(lvl) % 2 else [])
            got = simd.reduce_lanes(x, simd.ADD, partials=P)
            assert np.array_equal(got.view(np.uint32), lvl[0].view(np.uint32)), (nb, P)
        got = simd.reduce_lanes(x, simd.MAX, partials=3)
        assert np.array_equal(got, xf.max(axis=0)), nb
    # row mode: the per-lane results, then a balanced tree across the cfg's lanes
    lanes = simd.LANES
    x = (rng.standard_normal((5, lanes)) * 30).astype(F16)
    per_lane = simd.reduce_lanes(x, simd.ADD)
    lvl = [per_lane[i] for i in range(lanes)]
    while len(lvl) > 1:
        lvl = [(lvl[i] + lvl[i + 1]).astype(F32) for i in range(0, len(lvl) - 1, 2)] + \
            ([lvl[-1]] if len(lvl) % 2 else [])
    assert simd.reduce_row(x, simd.ADD) == simd.narrow16(lvl[0])


def test_int4_weights_round_trip_and_chunk():
    """An INT4 weight: values in [-7, 7] within half a step of W', unpack(blob) == q, and its
    first 64 columns are the blob's first K * 32 bytes (a streamed chunk)."""
    W, _ = _layer()
    p = pack("wq", W.wq, row_gain=W.in_norm, wbits=4)
    assert np.abs(p.q).max() <= 7
    want = W.wq * W.in_norm[:, None]
    assert np.all(np.abs(p.dequant() - want).max(axis=0) <= 0.5 * (1.0 + 1e-5) * p.s)
    blob = p.blob()
    assert blob.size == p.K * p.N // 2
    assert np.array_equal(unpack(blob, p.K, p.N, wbits=4), p.q)
    assert np.array_equal(blob[:p.K * 32], pack_int4(to_b_pairs(p.q[:, :64])))


def test_rope_orders_agree():
    d = model.load()
    cos, sin = model.rope_cos_sin(d, 4097)
    rng = np.random.default_rng(6)
    q, k = rng.normal(size=(2, d.q_rope))
    qi, ki = model.rope_interleaved(q, cos, sin), model.rope_interleaved(k, cos, sin)
    qh, kh = model.rope_hf(q, cos, sin), model.rope_hf(k, cos, sin)
    assert np.allclose(model.hf_to_interleaved(qh), qi)
    assert np.isclose(qi @ ki, qh @ kh), "the score must not depend on the stored pair order"


def test_yarn_constants():
    d = model.load()
    assert abs(d.softmax_scale - 0.1147) < 5e-5
    f = model.yarn_inv_freq(d)
    assert f.shape == (32,) and f[0] == 1.0 and np.all(np.diff(f) < 0)


def test_reference_absorbed_equals_expanded():
    W, _ = _layer()
    d = W.d
    rng = np.random.default_rng(7)
    cache = reference.build_cache(W, rng.normal(size=(15, d.hidden)).astype(F32), range(15))
    x = rng.normal(size=d.hidden).astype(F32)
    A = reference.attention(W, reference.rmsnorm(x, W.in_norm, d.rms_eps), 15, cache)
    assert np.allclose(A["s_lat"], A["scores"], rtol=1e-4, atol=1e-5)
    o = np.stack([A["o_lat"][h] @ W.w_uv(h) for h in range(d.heads)])
    assert np.allclose(o, A["o"], rtol=1e-3, atol=1e-6)


def test_kv_row_matches_the_layer_model():
    from . import golden, hwmodel
    g = golden.make(seed=1, L=15, n_calib=2, bc=16)
    row = hwmodel.kv_row(g.pack, g.scales, g.x16, g.pos)
    assert np.array_equal(row, np.concatenate([g.hw["c8_new"], g.hw["kpe8_new"]]))


def test_cache_copies_hold_their_elements():
    from .layout import key_copy, value_copy
    rng = np.random.default_rng(8)
    rows = rng.integers(-127, 128, size=(37, 576)).astype(np.int8)
    C = 64
    kc, vc = key_copy(rows, C), value_copy(rows[:, :512], C)
    for t in (0, 5, 16, 36):
        base = (t // 16) * 144 * 64 + (t % 16) * 4
        got = np.concatenate([kc[base + 64 * k: base + 64 * k + 4] for k in range(144)])
        assert np.array_equal(got, rows[t])
        vals = [vc[(i // 16) * C * 16 + (t // 4) * 64 + (i % 16) * 4 + t % 4] for i in range(512)]
        assert np.array_equal(np.array(vals, dtype=np.int8), rows[t, :512])


def test_fma32_rounds_once():
    from fractions import Fraction
    from .simd import fma32
    rng = np.random.default_rng(9)
    n = 3000
    a = (rng.standard_normal(n) * 2.0 ** rng.integers(-20, 20, n)).astype(F32)
    b = (rng.standard_normal(n) * 2.0 ** rng.integers(-20, 20, n)).astype(F32)
    c = (rng.standard_normal(n) * 2.0 ** rng.integers(-40, 40, n)).astype(F32)
    # a * b = 1 + 2^-11 + 2^-24, a float32 midpoint: +-2^-50 must decide, as the exact sum does
    a = np.append(a, [F32(1 + 2 ** -12)] * 3)
    b = np.append(b, [F32(1 + 2 ** -12)] * 3)
    c = np.append(c, [F32(0.0), F32(2 ** -50), F32(-2 ** -50)])
    got = fma32(a, b, c)
    for i in range(a.size):
        v = Fraction(float(a[i])) * Fraction(float(b[i])) + Fraction(float(c[i]))
        r = F32(float(v))
        cands = [np.nextafter(r, F32(-np.inf)), r, np.nextafter(r, F32(np.inf))]
        want = min(cands, key=lambda x: (abs(Fraction(float(x)) - v),
                                         int(np.asarray(x, dtype=F32).view(np.uint32)) & 1))
        assert got[i] == want, f"fma32({a[i]}, {b[i]}, {c[i]}) = {got[i]}, want {want}"


def test_simd_roms_match_the_generated_rtl():
    """simd.py's tables and constants against the SystemVerilog the RTL build generated (skipped
    when the split cluster's RTL has not been generated here)."""
    import re
    from . import simd
    sv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../generated/"
                      "snax_split_cluster_xdma/snax_split_cluster_blocks.sv")
    if not os.path.exists(sv):
        return
    text = open(sv).read()
    m = re.search(r"module FpActivation_exp32_silu256_rsq64\w*\((.*?)endmodule", text, re.S)
    assert m, "no FpActivation_exp32_silu256_rsq64 in the generated RTL"
    body = m.group(1)

    def rom(name, n):
        r = re.search(r"wire \[\d+:0\]\[31:0\]\s+%s =\s*'\{([^}]*)\}" % name, body)
        v = [int(x, 16) for x in re.findall(r"32'h([0-9A-Fa-f]+)", r.group(1))][::-1]
        return np.array(v[:n], dtype=np.uint32)

    def bits(t):
        return np.asarray(t, dtype=F32).view(np.uint32)

    # ROM order in the netlist: exp, silu slope, rsqrt slope, silu base, rsqrt base
    assert np.array_equal(rom("_GEN", simd.EXP_LUT_N), bits(simd.EXP_LUT))
    assert np.array_equal(rom("_GEN_0", simd.SILU_N), bits(simd.SILU_SLOPE))
    assert np.array_equal(rom("_GEN_1", simd.RSQ_N + 1), bits(simd.RSQ_SLOPE))
    assert np.array_equal(rom("_GEN_2", simd.SILU_N), bits(simd.SILU_BASE))
    assert np.array_equal(rom("_GEN_3", simd.RSQ_N + 1), bits(simd.RSQ_BASE))
    for c in (simd.HI_E, simd.LOG2EF_N, simd.LN2_N, simd.RSQ_SCALE, simd.RSQ_BIAS):
        assert f"32'h{int(bits(c)):08X}" in body, f"constant {float(c)} is not in the netlist"


def test_simd_activations_stay_within_an_ulp():
    from .fp import ulp16
    from .simd import EXP, RSQRT, SILU, stream_map
    x = np.arange(0, 0x7C00, dtype=np.uint16).view(F16)
    xs = np.concatenate([x, -x]).astype(np.float64)
    with np.errstate(over="ignore"):
        e_ref = np.exp(xs).astype(F16)
    fin = np.isfinite(e_ref) & (e_ref != 0)
    assert ulp16(stream_map(xs.astype(F16), 1.0, 0.0, EXP)[fin], e_ref[fin]).max() <= 1
    near = np.abs(xs) <= 10  # SILU's table ends at |x| = 16
    with np.errstate(over="ignore"):
        s_ref = (xs / (1 + np.exp(-xs))).astype(F16)
    assert ulp16(stream_map(xs.astype(F16), 1.0, 0.0, SILU)[near], s_ref[near]).max() <= 1
    pos = x[x > 0].astype(np.float64)
    r_ref = (1 / np.sqrt(pos)).astype(F16)
    ok = np.isfinite(r_ref)
    assert ulp16(stream_map(pos.astype(F16), 1.0, 0.0, RSQRT)[ok], r_ref[ok]).max() <= 1


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    fails = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001 -- a crash is a failure too, report and go on
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - fails}/{len(tests)} passed")
    return fails


if __name__ == "__main__":
    sys.exit(main())
