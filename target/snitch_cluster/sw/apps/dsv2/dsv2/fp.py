# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Bit-level numerics of snax_split_cluster, vectorised.

Every function here reproduces one piece of hardware arithmetic exactly, or states the
tolerance it holds to:

    d_port      the GEMM's Int32ToFp16Converter: RNE(acc * 2^-k), k clamped to 0..14,
                overflow to +-Inf. Exact.
    quant_i8    the SIMD's Fp16ToInt8: FP32 product with the FP32 inv_scale, clamp to
                [-128, 128], round ties-to-even, saturate to the SYMMETRIC [-127, 127].
                Exact.
    mul16/add16 one FP16 x FP16 -> FP16 SIMD stage: computed in FP32, rounded once. The
                FP32 product of two FP16 values is exact (11 x 11 significant bits), so
                mul16 is the correctly rounded FP16 product. Exact.
    bf16        round-to-nearest-even to bfloat16, the checkpoint's storage type.

The SIMD block's own FP32 arithmetic -- the map's fused multiply-add, the exp / silu /
rsqrt tables and the reduce's order of summation -- is simd.py's.
"""

import numpy as np

F16 = np.float16
F32 = np.float32
F64 = np.float64

FP16_MAX = 65504.0
D_SHIFT_MAX = 14  # the converter clamps k: 2^-14 is FP16's smallest normal number


def bits16(a):
    """FP16 values -> their uint16 bit patterns."""
    return np.array(a, dtype=F16).view(np.uint16)


def from_bits16(u):
    """uint16 bit patterns -> FP16 values."""
    return np.array(u, dtype=np.uint16).view(F16)


def f32bits(x):
    """One FP32 value -> its bit pattern, the form a SIMD scale CSR takes."""
    return int(np.asarray(x, dtype=F32).reshape(()).view(np.uint32))


def bf16(x):
    """Round to bfloat16 (ties to even), returned as float32 values."""
    u = np.ascontiguousarray(np.asarray(x, dtype=F32)).view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    u = ((u + 0x7FFF + lsb) >> 16) << 16
    return u.astype(np.uint32).view(F32)


def d_port(acc, k=0):
    """The D port's INT32 -> FP16: RNE(acc * 2^-k), k clamped to 0..14, overflow -> +-Inf.

    acc * 2^-k is exact in float64 (|acc| < 2^31), and numpy's float64 -> float16
    conversion rounds once, to nearest even, overflowing to Inf -- the converter's
    arithmetic. With k <= 14 no non-zero integer lands in the subnormals, so the one
    place the two could differ never arises. test_dsv2.py checks this against
    snax_utils.int32_to_fp16_golden, the RTL's scalar model.
    """
    a = np.asarray(acc, dtype=np.int64)
    if np.any(a > 2**31 - 1) or np.any(a < -2**31):
        raise OverflowError("an accumulator is outside INT32; the array would have wrapped")
    k = min(max(int(k), 0), D_SHIFT_MAX)
    with np.errstate(over="ignore"):
        return (a.astype(F64) * F64(2.0 ** -k)).astype(F16)


def d_shift_for_depth(K):
    """The smallest k with 127^2 * K <= 65,504 * 2^k: no INT8 GEMM of depth K can overflow."""
    k = 0
    while 127 * 127 * K > FP16_MAX * 2 ** k:
        k += 1
    if k > D_SHIFT_MAX:
        raise ValueError(f"depth {K} needs k = {k}, past the converter's {D_SHIFT_MAX}")
    return k


def quant_i8(x16, inv_scale):
    """Fp16ToInt8, bit for bit: sat_[-127,127](rne(clamp(fp32(x) * inv_scale, +-128)))."""
    v = np.asarray(x16, dtype=F16).astype(F32) * F32(inv_scale)
    v = np.clip(v, F32(-128.0), F32(128.0))
    return np.clip(np.rint(v), -127, 127).astype(np.int8)


def mul16(a, b):
    """One FP16 multiply stage: exact FP32 product, one RNE to FP16."""
    with np.errstate(over="ignore"):
        return (np.asarray(a, dtype=F16).astype(F32) * np.asarray(b, dtype=F16).astype(F32)
                ).astype(F16)


def add16(a, b):
    """One FP16 add stage: FP32 sum (exact for FP16 operands), one RNE to FP16."""
    with np.errstate(over="ignore"):
        return (np.asarray(a, dtype=F16).astype(F32) + np.asarray(b, dtype=F16).astype(F32)
                ).astype(F16)


def to16(x):
    """Round an FP32 intermediate to FP16 (the transport between SIMD stages)."""
    with np.errstate(over="ignore"):
        return np.asarray(x, dtype=F32).astype(F16)


def mono16(u):
    """FP16 bit patterns -> integers in the same order as the values they encode."""
    u = np.asarray(u, dtype=np.int64)
    mag = u & 0x7FFF
    return np.where(u & 0x8000, 0x8000 - mag, 0x8000 + mag)


def ulp16(a, b):
    """Distance in FP16 ULPs between two FP16 arrays (+0 and -0 are 1 apart)."""
    return np.abs(mono16(bits16(a)) - mono16(bits16(b)))
