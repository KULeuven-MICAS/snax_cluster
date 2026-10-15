#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data for snax-dsv2-moe-prefill: one cluster's share of HeMAiA's MoE prefill (the experts' gate|up
# phase of workloads/moe/common/moe_prefill_mc.py): E expert slots of `rows` padded token rows, each
# slot's intermediate columns in chunks of three 16-column blocks, every chunk a GEMM on array shape
# 3, (8, 8, 16), with INT4 weights, then the dequantisation and SwiGLU on the SIMD, then the slot's
# quantiser into a row-major INT8 A operand.
#
#   chunk   W [K, 2 nb 16] INT4 (the gate columns, then the up columns) and the scale tile of its
#           2 nb 16 output columns: FP16 factors, tiled over the eight rows of a D-layout row tile
#   D       RNE(X . W * 2^-k) in the D-layout                     the GEMM
#   g       RNE(D (.) scale)                                       dequantisation
#   a16     RNE(silu(gate) (.) up)                                 SwiGLU
#   a8      sat127(rne(a16 * inv)) over all of a slot's chunks     the quantiser
#
# The data is the same for every build: the three layouts of every chunk are linked and the app
# picks one (MP_LAYOUT). A chunk's image is what the ring load moves into a slab:
#
#   [lead: layout's B offset][INT4 weights][pad to 64 B][scale tile]
#
# A pool of POOL distinct chunk images serves all chunks (chunk i takes pool entry i % POOL), so
# the weights in DRAM stay small while every GEMM still reads real data; the golden follows.

import argparse
import pathlib
import sys

import hjson
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "snax-dsv2-gemm3-bench"
                       / "data"))
import g3data as g3  # noqa E402
from util.emit import Emitter  # noqa E402
from util.fp import F16, F32, f32bits, mul16, quant_i8, bits16  # noqa E402
from util.hwmodel import silu16  # noqa E402
from util.layout import to_d  # noqa E402
from util.fp import d_port  # noqa E402

BLK = 16        # a column block (Nu)
LAYS = ("b", "paired", "parity")


def chunk_list(blocks, per_chunk):
    """Blocks of each chunk of a slot: `per_chunk` each, the remainder last."""
    out = [per_chunk] * (blocks // per_chunk)
    if blocks % per_chunk:
        out.append(blocks % per_chunk)
    return out


def main():
    ap = argparse.ArgumentParser(description="Data for snax-dsv2-moe-prefill")
    ap.add_argument("--swcfg", type=pathlib.Path, required=True)
    ap.add_argument("--hwcfg", type=pathlib.Path, required=True)
    ap.add_argument("--header", type=pathlib.Path, required=True)
    ap.add_argument("--blob-dir", type=pathlib.Path, required=True)
    ap.add_argument("--set", default="unit", help="the workload: unit or full (params.hjson)")
    ap.add_argument("--rows", default="", help="override the set's slots' rows, e.g. 24,16,16,8")
    ap.add_argument("--blocks", type=int, default=0, help="override the set's blocks a slot")
    ap.add_argument("--boff", default="",
                    help="override B's offsets into its slab, e.g. b=64,paired=0")
    args = ap.parse_args()
    prm = hjson.loads(args.swcfg.read_text())
    g3.check_hwcfg(hjson.loads(args.hwcfg.read_text()))
    ws = dict(prm["sets"][args.set])
    if args.rows:
        ws["rows"] = [int(v) for v in args.rows.split(",")]
    if args.blocks:
        ws["blocks"] = args.blocks
    K, k = int(prm["K"]), int(prm["k"])
    per = int(prm["blocks_per_chunk"])
    rows, blocks, pool = ws["rows"], int(ws["blocks"]), int(ws["pool"])
    boff = {lay: int(prm["boff"][lay]) for lay in LAYS}
    for kv in filter(None, args.boff.split(",")):
        lay, v = kv.split("=")
        if lay not in LAYS or int(v) % 8:
            raise ValueError(f"--boff {kv}: a layout b|paired|parity, a multiple of 8")
        boff[lay] = int(v)
    if any(r % 8 or r < 8 for r in rows):
        raise ValueError(f"rows {rows}: whole 8-row tiles")
    ne = blocks                       # a slot's blocks: the cluster's half of an expert
    chunks = chunk_list(blocks, per)  # blocks of each chunk
    nbs = sorted(set(chunks), reverse=True)
    S, NC = len(rows), len(rows) * len(chunks)
    rng = np.random.default_rng(int(prm["seed"]))
    em = Emitter(args.blob_dir)
    em.c(f"// snax-dsv2-moe-prefill, set {args.set}: {S} slots of rows {rows}, {blocks} blocks "
         f"each in chunks {chunks}, seed {prm['seed']}.")
    for nm, v, c in (("MP_K", K, "the GEMMs' depth"), ("MP_KT", K // 8, "passes of K"),
                     ("MP_KSHIFT", k, "the D port's RNE(acc * 2^-k)"),
                     ("MP_NSLOT", S, "expert slots"), ("MP_NCHUNK", NC, "chunks in all"),
                     ("MP_CPS", len(chunks), "chunks a slot"), ("MP_NE", ne, "blocks a slot"),
                     ("MP_NBMAX", max(chunks), "blocks of the largest chunk"),
                     ("MP_ROWMAX", max(rows), "rows of the largest slot"),
                     ("MP_POOL", pool, "distinct chunk images")):
        em.define(nm, v, c)
    for lay in LAYS:
        em.define(f"MP_BOFF_{lay.upper()}", boff[lay], f"B's offset into its slab, layout {lay}")
    em.define("MP_PX", g3.pitch(K), "X rows' pitch in L1")
    em.define("MP_PE", g3.pitch(ne * BLK), "a8 rows' pitch in L1")

    # ---- the pool: weights, scale factors; their D-layout images in every layout ----------------
    W, F, img = {}, {}, {}
    for p in range(pool):
        for nb in nbs:
            n = 2 * nb * BLK
            W[p, nb] = rng.integers(-8, 8, size=(K, n)).astype(np.int8)
            F[p, nb] = None
    # the slots' X rows
    X = [rng.integers(-128, 128, size=(r, K)).astype(np.int8) for r in rows]
    # scale factors: bring the dequantised values to about 1
    d0 = d_port(X[0][:8].astype(np.int64) @ W[0, nbs[0]].astype(np.int64), k).astype(F32)
    sigma = float(d0.std())
    for key in W:
        n = W[key].shape[1]
        F[key] = (rng.uniform(0.5, 1.5, size=n) / sigma).astype(F16)

    def scale_tile(f):
        return bits16(to_d(np.tile(np.asarray(f, dtype=F16), (8, 1)), g3.MESH3))

    for (p, nb), w in W.items():
        for lay in LAYS:
            wb = np.asarray(g3.b_blob(w, lay)).view(np.uint8)
            lead = np.zeros(boff[lay], dtype=np.uint8)
            body = np.concatenate([lead, wb])
            soff = -(-body.size // 64) * 64
            st = scale_tile(F[p, nb]).view(np.uint8)
            image = np.concatenate([body, np.zeros(soff - body.size, dtype=np.uint8), st])
            name = em.blob(f"mp_c{p}_n{nb}_{lay}", image.view(np.int8))
            img[p, nb, lay] = (name, image.size, soff)

    # ---- the golden: every slot's chunks, then its quantiser -------------------------------------
    a16 = [np.zeros((r, ne * BLK), dtype=F16) for r in rows]
    for s in range(S):
        b0 = 0
        for j, nb in enumerate(chunks):
            p = (s * len(chunks) + j) % pool
            n = nb * BLK
            D = d_port(X[s].astype(np.int64) @ W[p, nb].astype(np.int64), k)
            if np.any(np.isinf(D.astype(F32))):
                raise ValueError("a GEMM output overflows FP16")
            g = mul16(D, np.asarray(F[p, nb], dtype=F16)[None, :])
            sg = silu16(g[:, :n])
            a16[s][:, b0 * BLK:(b0 + nb) * BLK] = mul16(sg, g[:, n:])
            b0 += nb
    peak = max(float(np.abs(a.astype(F32)).max()) for a in a16)
    inv = 1.0
    while peak * inv * 2 <= 127.0:
        inv *= 2.0
    while peak * inv > 127.0:
        inv /= 2.0
    a8 = [quant_i8(a, inv) for a in a16]
    sat = sum(int((np.abs(a) == 127).sum()) for a in a8)
    em.define("MP_INV_BITS", f"0x{f32bits(inv):08X}u", f"the quantiser's 1/scale = {inv}")
    em.blob("mp_gold_a8", np.concatenate([a.reshape(-1) for a in a8]))

    # ---- the tables -----------------------------------------------------------------------------
    gold_off, xs = 0, []
    srows = []
    for s in range(S):
        xs.append(em.blob(f"mp_x{s}", X[s].reshape(-1)))
        xa = {al: em.blob(f"mp_xa{s}_{al}", g3.a_blob(X[s], al, 0)) for al in ("blocked", "ilv")}
        srows.append(f"    {{{rows[s]}u, {xs[s]}, {xa['blocked']}, {xa['ilv']}, {gold_off}u}},")
        gold_off += rows[s] * ne * BLK
    em.c("typedef struct {\n    uint32_t rows;       // padded token rows\n"
         "    const int8_t *x;     // the rows of X, K bytes each\n"
         "    const int8_t *xb, *xi;  // X in A-layout tiles, and with the passes interleaved\n"
         "    uint32_t gold;       // the slot's a8 in mp_gold_a8, rows x NE x 16 bytes\n"
         "} mp_slot_t;")
    em.c("static const mp_slot_t mp_slots[MP_NSLOT] = {\n" + "\n".join(srows) + "\n};")
    em.c("typedef struct {\n    const int8_t *img;   // the chunk image in DRAM\n"
         "    uint32_t bytes;      // its size\n"
         "    uint32_t soff;       // the scale tile's offset in it\n"
         "    uint32_t nb;         // gate (and up) blocks\n"
         "    uint32_t b0;         // first block in the slot\n"
         "} mp_chunk_t;")
    for lay in LAYS:
        crows = []
        for s in range(S):
            b0 = 0
            for j, nb in enumerate(chunks):
                p = (s * len(chunks) + j) % pool
                name, size, soff = img[p, nb, lay]
                crows.append(f"    {{{name}, {size}u, {soff}u, {nb}u, {b0}u}},")
                b0 += nb
        em.c(f"static const mp_chunk_t mp_chunks_{lay}[MP_NCHUNK] = {{\n" + "\n".join(crows)
             + "\n};")
    em.define("MP_IMGMAX", max(v[1] for v in img.values()), "the largest chunk image")
    em.write(args.header, args.blob_dir / "blobs.S")
    macs = sum(r // 8 * K // 8 * 2 * ne for r in rows)
    sys.stderr.write(f"[moe-prefill datagen] set {args.set}: {S} slots {rows}, {NC} chunks, "
                     f"{em.bytes / 2**20:.2f} MiB of blobs, {macs} array passes, a16 peak "
                     f"{peak:.2f} inv {inv}, {sat} saturated of {gold_off}\n")


if __name__ == "__main__":
    main()
