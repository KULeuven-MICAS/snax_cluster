# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""The SIMD block's FP32 arithmetic, bit for bit: StreamMap with its FpActivation, and StreamReduce,
as hw/chisel builds them for snax_split_cluster.

Every FP unit in that datapath is a port of fpnew (IEEE binary32, round to nearest even,
subnormals kept), so a numpy float32 add or multiply is an exact model of one. The fused
multiply-add rounds once; fma32 below emulates it exactly. The transcendental functions are
not libm calls but small tables read by the same FP units, so they too are reproduced exactly
here, table by table and operation by operation:

    StreamMap     t = fma(a, x, b)                           FP32 a, b; x the FP16 lane
                  act(t): LINEAR, EXP, SILU or RSQRT         FpActivation, below
                  narrow to FP16                              IEEE RNE (FpHelpers.narrow)

    FpActivation  s0 = fma(pre, scale, bias)                  the argument onto a node index
                  i, frac = round(s0), s0 - round(s0)         the magic-number round, FP32 adds
                  EXP    2^(i mod 32 / 32) * fma(frac, ln2/32, 1) * 2^(i div 32)
                  SILU   g = fma(frac, slope[i], base[i]),    g ~ sigmoid(-|t|), 256 nodes over
                         (t < 0 ? g : 1 - g) * t              [0, 16]
                  RSQRT  t = 2^e m, m' = 2^p m in [1, 4):     65 tangent nodes over [1, 4],
                         fma(frac, slope[i], base[i]) * 2^-(e-p)/2

    StreamReduce  per lane, four rotating FP32 partials (beat k into partial k mod 4, each an
                  FMA accumulate: x*1 + acc for ADD, x*x + acc for SUMSQ), folded
                  ((p0 + p1) + p2) + p3; then, unless LANEWISE, a pairwise tree over the 32
                  lanes; narrow to FP16.

The tables are computed as the Scala generator computes them (a double, rounded once to
float); test_dsv2 checks them against the generated SystemVerilog when it is present.
"""

import math

import numpy as np

F16 = np.float16
F32 = np.float32
F64 = np.float64

# StreamMap's ROM depths (hw/chisel StreamMap: expLutN, siluN, rsqN)
EXP_LUT_N = 32
SILU_N = 256
RSQ_N = 64
# StreamReduce accPartials in cfg/snax_split_cluster.hjson
REDUCE_PARTIALS = 4

LINEAR, EXP, SILU, RSQRT = 0, 1, 2, 3  # the StreamMap func CSR encoding


def _c(x):
    """A generator constant: a Scala double rounded once to float."""
    return F32(x)


# ---- FpActivation's constants and ROMs -----------------------------------------------------------

MAGIC = _c(12582912.0)                              # 1.5 * 2^23: float -> integer round
LOG2EF_N = _c(1.44269504088896341 * EXP_LUT_N)
LN2_N = _c(0.6931471805599453 / EXP_LUT_N)
HI_E = F32(np.float32("88.3762626647949"))
LO_E = -HI_E
EXP_LUT = np.array([_c(math.pow(2.0, i / EXP_LUT_N)) for i in range(EXP_LUT_N)], dtype=F32)

_SILU_XHI = 16.0
_SILU_H = _SILU_XHI / SILU_N
SILU_SCALE = _c(1.0 / _SILU_H)
SILU_HI = _c(_SILU_XHI)


def _gnode(i):
    xx = -(i * _SILU_H)
    return 1.0 / (1.0 + math.exp(-xx))


SILU_BASE = np.array([_c(_gnode(i)) for i in range(SILU_N)], dtype=F32)
SILU_SLOPE = np.array([_c((_gnode(i + 1) - _gnode(i - 1)) / 2.0) for i in range(SILU_N)],
                      dtype=F32)

_RSQ_H = 3.0 / RSQ_N
RSQ_SCALE = _c(1.0 / _RSQ_H)
RSQ_BIAS = _c(-1.0 / _RSQ_H)
RSQ_BASE = np.array([_c(1.0 / math.sqrt(1.0 + i * _RSQ_H)) for i in range(RSQ_N + 1)], dtype=F32)
RSQ_SLOPE = np.array([_c(-0.5 * math.pow(1.0 + i * _RSQ_H, -1.5) * _RSQ_H)
                      for i in range(RSQ_N + 1)], dtype=F32)


# ---- FP32 primitives ------------------------------------------------------------------------------

def fma32(a, b, c):
    """IEEE binary32 a * b + c with ONE round to nearest even, elementwise.

    The product is exact in float64 (24 x 24 significant bits) and the float64 sum s carries its
    exact rounding error e (TwoSum), so s + e == a b + c. Rounding s to float32 is then correct
    except when s lands exactly on a float32 midpoint the exact sum is not on; e says which way.
    """
    a64 = np.asarray(a, dtype=F32).astype(F64)
    b64 = np.asarray(b, dtype=F32).astype(F64)
    c64 = np.asarray(c, dtype=F32).astype(F64)
    with np.errstate(over="ignore", invalid="ignore"):
        p = a64 * b64
        s = p + c64
        bb = s - p
        e = (p - (s - bb)) + (c64 - bb)
        r = s.astype(F32)
        up = np.nextafter(r, F32(np.inf))
        dn = np.nextafter(r, F32(-np.inf))
        other = np.where(r.astype(F64) > s, dn, up)
        at_mid = np.isfinite(s) & np.isfinite(r) & (e != 0) & \
            (s == (r.astype(F64) + other.astype(F64)) * 0.5)
    fixed = np.where(e > 0, np.maximum(r, other), np.minimum(r, other))
    return np.where(at_mid, fixed, r).astype(F32)


def fp32max(a, b):
    """FpHelpers.fp32max: a unless b is larger, compared on the bit patterns (ties keep a)."""
    ua = np.asarray(a, dtype=F32).view(np.uint32)
    ub = np.asarray(b, dtype=F32).view(np.uint32)
    sa, sb = ua >> 31, ub >> 31
    keep_a = np.where(sa != sb, sa == 0, np.where(sa == 1, ub >= ua, ua >= ub))
    return np.where(keep_a, np.asarray(a, dtype=F32), np.asarray(b, dtype=F32)).astype(F32)


def narrow16(x):
    """FP32 -> FP16, IEEE round to nearest even, overflow to +-Inf (FpHelpers.narrow)."""
    with np.errstate(over="ignore"):
        return np.asarray(x, dtype=F32).astype(F16)


# ---- FpActivation ---------------------------------------------------------------------------------

def _round_split(s0):
    rM = (s0 + MAGIC).astype(F32)
    roundF = (rM - MAGIC).astype(F32)
    frac = (s0 - roundF).astype(F32)
    iM = rM.view(np.int32).astype(np.int64) - 0x4B400000
    return frac, iM


def activation(t, func):
    """FpActivation on FP32 inputs: EXP, SILU or RSQRT, FP32 out."""
    t = np.ascontiguousarray(np.asarray(t, dtype=F32))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if func == EXP:
            pre = np.maximum(np.minimum(t, HI_E), LO_E)
            s0 = fma32(pre, LOG2EF_N, F32(0.0))
            frac, iM = _round_split(s0)
            interp = fma32(frac, LN2_N, F32(1.0))
            r3 = (EXP_LUT[iM & (EXP_LUT_N - 1)] * interp).astype(F32)
            nE = iM >> int(math.log2(EXP_LUT_N))
            pow2 = np.where(nE < -127, 0, ((nE + 127) & 0xFF) << 23).astype(np.uint32).view(F32)
            return (r3 * pow2).astype(F32)
        if func == SILU:
            pre = np.minimum(np.abs(t), SILU_HI)
            s0 = fma32(pre, SILU_SCALE, F32(0.0))
            frac, iM = _round_split(s0)
            idx = np.where(iM > SILU_N - 1, SILU_N - 1, iM & (SILU_N - 1))
            g = fma32(frac, SILU_SLOPE[idx], SILU_BASE[idx])
            r3 = np.where(np.signbit(t), g, (F32(1.0) - g).astype(F32))
            return (r3 * t).astype(F32)
        if func == RSQRT:
            u = t.view(np.uint32).astype(np.int64)
            ex = (u >> 23) & 0xFF
            p = (~ex) & 1
            mprime = (((127 + p) << 23) | (u & 0x7FFFFF)).astype(np.uint32).view(F32)
            nR = -((ex - 127 - p) >> 1)
            efield = (nR + 127) & 0xFF
            flush = ((u >> 31) == 1) | (ex == 0) | (ex == 255)
            s0 = fma32(mprime, RSQ_SCALE, RSQ_BIAS)
            frac, iM = _round_split(s0)
            idx = np.where(iM > RSQ_N, RSQ_N, iM & 127)
            r3 = fma32(frac, RSQ_SLOPE[idx], RSQ_BASE[idx])
            pow2 = np.where(flush, 0, efield << 23).astype(np.uint32).view(F32)
            return (r3 * pow2).astype(F32)
    raise ValueError(f"FpActivation has no function {func}")


# ---- StreamMap ------------------------------------------------------------------------------------

def stream_map(x16, a=1.0, b=0.0, func=LINEAR):
    """One StreamMap pass: narrow(act(fma(a, x, b))) on FP16 lanes."""
    x = np.asarray(x16, dtype=F16).astype(F32)
    t = fma32(np.full(x.shape, a, dtype=F32), x, np.full(x.shape, b, dtype=F32))
    return narrow16(t if func == LINEAR else activation(t, func))


# ---- StreamReduce ---------------------------------------------------------------------------------

ADD, SUMSQ, MAX = "add", "sumsq", "max"


def _acc(op, v, acc):
    if op == MAX:
        return v if acc is None else fp32max(v, acc)
    y = v if op == SUMSQ else np.ones_like(v)
    return fma32(v, y, np.zeros_like(v) if acc is None else acc)


def reduce_lanes(beats16, op, partials=REDUCE_PARTIALS):
    """The per-lane FP32 result of a reduce over beats16 [n_beats, lanes] (FP16): partial k mod P
    accumulates beat k, the partials fold left in order, an unused partial is the identity."""
    x = np.asarray(beats16, dtype=F16).astype(F32)
    parts = []
    for p in range(min(partials, x.shape[0])):
        acc = None
        for k in range(p, x.shape[0], partials):
            acc = _acc(op, x[k], acc)
        parts.append(acc)
    tot = parts[0]
    for q in parts[1:]:
        tot = fp32max(tot, q) if op == MAX else (tot + q).astype(F32)
    if op != MAX and len(parts) < partials:
        tot = (tot + F32(0.0)).astype(F32)  # the masked partials add +0 (turns -0 into +0)
    return tot


def reduce_lanewise(beats16, op):
    """StreamReduce LANEWISE: one FP16 beat of per-lane results."""
    return narrow16(reduce_lanes(beats16, op))


def reduce_row(beats16, op):
    """StreamReduce row mode: the per-lane results folded pairwise across lanes, one FP16 scalar."""
    v = reduce_lanes(beats16, op)
    while v.shape[-1] > 1:
        a, b = v[..., 0::2], v[..., 1::2]
        v = fp32max(a, b) if op == MAX else (a + b).astype(F32)
    return narrow16(v[..., 0])


def as_beats(row16, lanes=32):
    """A 1-D FP16 row as the beats the reduce reads: [len / lanes, lanes]."""
    r = np.asarray(row16, dtype=F16).reshape(-1)
    if r.size % lanes:
        raise ValueError(f"a row of {r.size} does not fill beats of {lanes} lanes")
    return r.reshape(-1, lanes)
