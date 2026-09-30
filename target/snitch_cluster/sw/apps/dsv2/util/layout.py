# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""The VersaCore array's operand layouts, and the GEMV forms built on them.

With the spatial unrolling (Mu, Ku, Nu) = (meshRow, tileSize, meshCol):

    A of X [M, K]   X[m*Mu + r, k*Ku + s]  at  ((m*K_T + k)*Mu + r)*Ku + s
    B of Y [K, N]   Y[k*Ku + s, n*Nu + c]  at  ((n*K_T + k)*Nu + c)*Ku + s
    D of Z [M, N]   Z[m*Mu + r, n*Nu + c]  at  ((m*N_T + n)*Mu + r)*Nu + c

with K_T = K / Ku and N_T = N / Nu. These are the orders block_gemm_golden_model() in
util/sim/snax_utils.py assumes and the streamers walk.

A GEMV is a GEMM with one real row: x fills row 0 of a 16-row A operand whose rows
1..15 are zero. Its weight streams as CHUNKS of whole output columns: chunk j is
columns [j*C, (j+1)*C) of W in B-layout, K*C bytes, so one GEMM task per chunk yields
C finished outputs and nothing accumulates across tasks.

INT4 WEIGHTS go through the B reader's converter, which widens the LOW half of a B beat,
nibble i to byte i. A (1, 4, 32) pass takes two Nu-column blocks at one k, and the low
half of the beat is the reader's channels 0..7: eight consecutive words. So the two blocks
of a pass must sit together, which is the B-layout with blocks of 2 Nu columns
(to_b_pairs), nibble-packed (pack_int4): 64 contiguous bytes a pass. Being n-major like
to_b, any multiple of 2 Nu columns is still a byte slice of it.
"""

import numpy as np

MESH = (16, 4, 16)  # snax_split_cluster's VersaCore: (Mu, Ku, Nu)
GEMV_CHUNK = 64     # output columns per streamed weight chunk: 4 n-blocks of Nu


def mesh_from_hwcfg(hw):
    """(Mu, Ku, Nu) of array shape 0, whose blocks the operand layouts use, from a parsed cluster
    cfg of one data type. Every other shape must be a GEMV unrolling with the same Ku, at most
    Mu rows and a multiple of that Nu: it reads the first rows of the same layouts (snax-dsv2.h,
    THE GROUPED GEMV)."""
    acc = hw["snax_versacore_core_template"]["snax_acc_cfg"][0]
    unrolling = acc["snax_versacore_spatial_unrolling"]
    if len(unrolling) != 1:
        raise ValueError("the kernels program data_type = 0; the cfg declares more data types")
    mesh = tuple(int(v) for v in unrolling[0][0])
    for shape in unrolling[0][1:]:
        mu, ku, nu = (int(v) for v in shape)
        if mu > mesh[0] or ku != mesh[1] or nu % mesh[2]:
            raise ValueError(f"array shape {(mu, ku, nu)} does not read shape 0's layouts {mesh}")
    return mesh


def to_a(R, mesh=MESH):
    """A-layout of R [M, K], flat."""
    mr, ts, _ = mesh
    M, K = R.shape
    return np.ascontiguousarray(R.reshape(M // mr, mr, K // ts, ts).transpose(0, 2, 1, 3)).reshape(-1)


def from_a(flat, M, K, mesh=MESH):
    mr, ts, _ = mesh
    return np.ascontiguousarray(
        np.asarray(flat).reshape(M // mr, K // ts, mr, ts).transpose(0, 2, 1, 3)).reshape(M, K)


def to_b(R, mesh=MESH):
    """B-layout of R [K, N], flat."""
    _, ts, mc = mesh
    K, N = R.shape
    return np.ascontiguousarray(R.reshape(K // ts, ts, N // mc, mc).transpose(2, 0, 3, 1)).reshape(-1)


def from_b(flat, K, N, mesh=MESH):
    _, ts, mc = mesh
    return np.ascontiguousarray(
        np.asarray(flat).reshape(N // mc, K // ts, mc, ts).transpose(1, 3, 0, 2)).reshape(K, N)


def to_b_pairs(R, mesh=MESH):
    """B-layout of R [K, N] in blocks of 2 Nu columns: the weights of a (1, Ku, 2 Nu) pass, in
    the order the array reads them from one B beat."""
    mr, ts, mc = mesh
    return to_b(R, (mr, ts, 2 * mc))


def from_b_pairs(flat, K, N, mesh=MESH):
    mr, ts, mc = mesh
    return from_b(flat, K, N, (mr, ts, 2 * mc))


def pack_int4(q):
    """int8 values in [-8, 7], two to a byte, value 2i in the low nibble of byte i: what the B
    reader's converter sign-extends back, nibble i into byte i."""
    q = np.asarray(q).reshape(-1)
    if q.size % 2 or np.any(q < -8) or np.any(q > 7):
        raise ValueError("INT4 packing takes an even count of values in [-8, 7]")
    u = (q.astype(np.int16) & 0xF).astype(np.uint8)
    return (u[0::2] | (u[1::2] << 4)).view(np.int8)


def unpack_int4(b):
    """Inverse of pack_int4: nibble i sign-extended, as the converter does."""
    u = np.asarray(b).reshape(-1).view(np.uint8)
    lo, hi = (u & 0xF).astype(np.int8), (u >> 4).astype(np.int8)
    out = np.empty(2 * u.size, dtype=np.int8)
    out[0::2] = np.where(lo > 7, lo - 16, lo)
    out[1::2] = np.where(hi > 7, hi - 16, hi)
    return out


def to_d(R, mesh=MESH):
    """D-layout of R [M, N], flat."""
    mr, _, mc = mesh
    M, N = R.shape
    return np.ascontiguousarray(R.reshape(M // mr, mr, N // mc, mc).transpose(0, 2, 1, 3)).reshape(-1)


def gemv_a(x, mesh=MESH):
    """A-layout of the [Mu, K] matrix with row 0 = x, rows 1.. = 0."""
    mr, _, _ = mesh
    R = np.zeros((mr, x.shape[-1]), dtype=x.dtype)
    R[0] = x
    return to_a(R, mesh)


def gemv_a_rep(x, mesh=MESH):
    """A GEMV's A operand with x in all Mu rows, so every row of the output is x @ W: what
    snax-dsv2-mla-fa's kernel writes for the W_UV GEMVs. A one-token GEMV reads row 0 (several
    tokens use a compact four-row operand, token t in row t); the apps' inputs are the vectors
    alone, and the quantiser writes those rows only (snax-dsv2.h, dsv2_quant_at)."""
    mr, _, _ = mesh
    return to_a(np.repeat(np.asarray(x)[None, :], mr, axis=0), mesh)


def to_a4(xs):
    """The compact A operand of up to four tokens (snax-dsv2.h, SEVERAL TOKENS), flat: token t's
    x in row t, the other rows zero; A-block k (row r's four values at byte 4 r) at
    64 (k % 8) + 16 ((k // 8) % 4) + 512 (k // 32), so consecutive blocks are 64 bytes apart."""
    xs = np.asarray(xs, dtype=np.int8)
    T, K = xs.shape
    if T > 4 or K % 128:
        raise ValueError(f"[{T}, {K}]: at most four tokens, K a multiple of 128")
    k = np.arange(K // 4)
    base = 64 * (k % 8) + 16 * ((k // 8) % 4) + 512 * (k // 32)
    out = np.zeros(4 * K, dtype=np.int8)
    for t in range(T):
        for i in range(4):
            out[base + 4 * t + i] = xs[t, 4 * k + i]
    return out


def b_chunks(W, chunk=GEMV_CHUNK, mesh=MESH):
    """W [K, N] as consecutive B-layout chunks of `chunk` output columns, flat."""
    K, N = W.shape
    if N % chunk:
        raise ValueError(f"N = {N} is not a multiple of the chunk width {chunk}")
    return np.concatenate([to_b(W[:, j:j + chunk], mesh) for j in range(0, N, chunk)])


def from_b_chunks(flat, K, N, chunk=GEMV_CHUNK, mesh=MESH):
    flat = np.asarray(flat).reshape(N // chunk, K * chunk)
    return np.concatenate([from_b(c, K, chunk, mesh) for c in flat], axis=1)


# ---- the latent cache, two copies in DRAM ---------------------------------------------------
#
# KEY copy: A-layout of [capacity, 576] (tokens x [c | k_pe]), blocks of 16 tokens x 4 values.
#   A tile of Bc tokens is contiguous: bytes [j Bc 576, (j+1) Bc 576). Appending token t writes
#   144 runs of 4 bytes, 64 bytes apart, from (t / 16) * 144 * 64 + (t % 16) * 4.
# VALUE copy: A-layout of V^T = [512, capacity] (latents x tokens), blocks of 16 latents x 4
#   tokens. Element (i, t) sits at (i / 16) * capacity * 16 + (t / 4) * 64 + (i % 16) * 4 + t % 4,
#   so a tile is 32 runs of 16 Bc bytes and an append is 512 runs of one byte.

def key_copy(rows, capacity, mesh=MESH):
    """rows [T, 576] int8 -> the key copy, tokens past T zero."""
    K = np.zeros((capacity, rows.shape[1]), dtype=np.int8)
    K[:rows.shape[0]] = rows
    return to_a(K, mesh)


def value_copy(c8, capacity, mesh=MESH):
    """c8 [T, 512] int8 -> the value copy, tokens past T zero."""
    V = np.zeros((c8.shape[1], capacity), dtype=np.int8)
    V[:, :c8.shape[0]] = c8.T
    return to_a(V, mesh)
