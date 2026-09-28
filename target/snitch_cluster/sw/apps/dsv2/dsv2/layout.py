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
"""

import numpy as np

MESH = (16, 4, 16)  # snax_split_cluster's VersaCore: (Mu, Ku, Nu)
GEMV_CHUNK = 64     # output columns per streamed weight chunk: 4 n-blocks of Nu


def mesh_from_hwcfg(hw):
    """(Mu, Ku, Nu) of array shape 0, whose blocks the operand layouts use, from a parsed cluster
    cfg of one data type. Every other shape must be a one-row GEMV unrolling with the same Ku and
    a multiple of that Nu: it reads the same layouts (snax-dsv2.h, THE GROUPED GEMV)."""
    acc = hw["snax_versacore_core_template"]["snax_acc_cfg"][0]
    unrolling = acc["snax_versacore_spatial_unrolling"]
    if len(unrolling) != 1:
        raise ValueError("the kernels program data_type = 0; the cfg declares more data types")
    mesh = tuple(int(v) for v in unrolling[0][0])
    for shape in unrolling[0][1:]:
        mu, ku, nu = (int(v) for v in shape)
        if mu != 1 or ku != mesh[1] or nu % mesh[2]:
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
    snax-dsv2-mla-fa's kernel writes for the W_UV GEMVs. A GEMV reads row 0 (and row 4 for a
    second token); the apps' inputs are the vectors alone, and the quantiser writes those rows
    only (snax-dsv2.h, dsv2_quant_a2)."""
    mr, _, _ = mesh
    return to_a(np.repeat(np.asarray(x)[None, :], mr, axis=0), mesh)


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
