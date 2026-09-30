#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data for snax-dsv2-gemv: the eight GEMV shapes of DeepSeek-V2-Lite layer 1, with the
# layer's own tensors. Every input, weight and expected output comes from one golden pack
# (sw/apps/dsv2/util), so each shape runs on the activation that really reaches it in the
# layer -- the quantised normed input for W_Q, the quantised SwiGLU output for an expert's
# down projection -- and "passes here" means "passes inside the layer".
#
# Per shape the app gets, all in DRAM, for GEMV_DATA_NTOK = 4 tokens:
#   <s>_a    x (INT8, [4, K]), the vectors alone; a one-token run places token 0 in row 0 of the
#            A buffer. Token 0 is the activation the layer computes; tokens 1..3 are fixed
#            permutations of it, so they carry the same values in other columns.
#   <s>_a4   the four tokens as the compact A operand of several tokens (util/layout.py to_a4),
#            4 K bytes, which a two- or four-token run loads as it is
#   <s>_w    the weight, [K, N] in B-layout: INT8, chunk j at bytes [j*K*64, (j+1)*K*64); or,
#            with --wbits 4, INT4 in the paired layout (util/layout.py), K*32 bytes a chunk.
#            The router is INT8 either way.
#   <s>_s    the per-column dequantisation factor s_x * s_w[n] * 2^k, FP16, one for all tokens
#   <s>_y    expected RNE(x.W * 2^-k), FP16, [4, N] (the D port's output)
#   <s>_yd   expected y (.) s, FP16, [4, N] (the dequantised output)
# and for the router, the D port's output at k = 0 ([4, 64]): the negative control, which must
# hold +-Inf wherever |x.W| rounds past 65,504. k is the worst case for depth K, so no INT8
# token overflows at it.

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))  # the util package
from util import golden  # noqa E402
from util.emit import Emitter  # noqa E402
from util.fp import bits16, d_port, mul16  # noqa E402
from util.layout import GEMV_CHUNK, mesh_from_hwcfg, to_a4  # noqa E402

# (key, printed name) in run order. The router is the negative control's shape: one chunk.
SHAPES = [
    ("wq", "W_Q"),
    ("wdkv", "W_DKV"),
    ("wo", "W_O"),
    ("router", "router"),
    ("expert_gu", "expert gate|up"),
    ("expert_down", "expert down"),
    ("shared_gu", "shared gate|up"),
    ("shared_down", "shared down"),
]


def shape_data(g):
    """key -> (x int8 [K], packed weight, k, y_raw, s, yd) from the golden pack."""
    H, P = g.hw, g.pack
    ks = H["ks"]
    e0 = int(H["ids"][0])
    gu0, dn0 = P.expert(e0)
    s0, sh = H["slots"][0], H["shared"]
    return {
        "wq": (H["xq"], P.wq, ks["x"], H["q_raw"], H["q_s"], H["q16"]),
        "wdkv": (H["xq"], P.wdkv, ks["x"], H["kv_raw"], H["kv_s"], H["ckv16"]),
        "wo": (H["o8"], P.wo, ks["x"], H["a_raw"], H["a_s"], H["attn16"]),
        "router": (H["hq"], P.wr, ks["x"], H["lg_raw"], H["lg_s"], H["logits16"]),
        "expert_gu": (H["hq"], gu0, ks["x"], s0["g_raw"], s0["g_s"], s0["g16"]),
        "expert_down": (s0["a8"], dn0, ks["ed"], s0["y_raw"], s0["y_s"], s0["y16"]),
        "shared_gu": (H["hq"], P.shared_gu, ks["x"], sh["g_raw"], sh["g_s"], sh["g16"]),
        "shared_down": (sh["a8"], P.shared_down, ks["sd"], sh["y_raw"], sh["y_s"], sh["y16"]),
    }


NTOK = 4  # the tokens a (4, 4, 32) pass serves


def tokens(key, x):
    """[NTOK, K]: x, then NTOK - 1 fixed permutations of it (seeded by the shape)."""
    rng = np.random.default_rng(sum(map(ord, key)))
    x = np.asarray(x, dtype=np.int8)
    return np.stack([x] + [rng.permutation(x) for _ in range(NTOK - 1)])


def main():
    ap = argparse.ArgumentParser(description="Data for snax-dsv2-gemv")
    ap.add_argument("--swcfg", type=pathlib.Path, required=True)
    ap.add_argument("--hwcfg", type=pathlib.Path, required=True)
    ap.add_argument("--header", type=pathlib.Path, required=True)
    ap.add_argument("--blob-dir", type=pathlib.Path, required=True)
    ap.add_argument("--wbits", type=int, default=8, choices=(8, 4),
                    help="weight width: INT8, or INT4 through the B converter")
    args = ap.parse_args()
    prm = hjson.loads(args.swcfg.read_text())
    mesh = mesh_from_hwcfg(hjson.loads(args.hwcfg.read_text()))
    if mesh != (16, 4, 16):
        raise ValueError(f"the kernel's descriptors assume a (16, 4, 16) mesh, not {mesh}")
    chunk = int(prm.get("chunk", GEMV_CHUNK))
    if chunk != GEMV_CHUNK:
        raise ValueError(f"the packer's chunk is {GEMV_CHUNK} columns")
    keys = prm.get("shapes", [k for k, _ in SHAPES])

    g = golden.make(seed=int(prm["seed"]), L=int(prm["L"]), wbits=args.wbits)
    data = shape_data(g)
    names = dict(SHAPES)

    em = Emitter(args.blob_dir)
    em.define("DSV2_DATA_WBITS", g.wbits, "the weights' width; the kernel's DSV2_WBITS must match")
    em.c(f"// DeepSeek-V2-Lite layer 1, golden seed {g.seed}, L = {g.L}; the expert shapes use"
         f" expert {int(g.hw['ids'][0])},\n// the golden token's first pick.")
    em.define("GEMV_CHUNK", chunk, "output columns per streamed weight chunk")
    em.define("GEMV_DATA_NTOK", NTOK, "tokens per shape: x, y and yd are [NTOK, K or N]")
    em.define("GEMV_NSHAPES", len(keys))
    kmax = max(data[k][1].K for k in keys)
    nmax = max(data[k][1].N for k in keys)
    em.define("GEMV_KMAX", kmax)
    em.define("GEMV_NMAX", nmax)
    rows = []
    neg = -1
    for i, key in enumerate(keys):
        x, p, k, y, s, yd = data[key]
        if p.N % chunk or p.K % 64:
            raise ValueError(f"{key}: [{p.K}, {p.N}] does not tile into {chunk}-column chunks")
        xs = tokens(key, x)
        ys, yds = [], []
        for t in range(NTOK):
            yt = d_port(xs[t].astype(np.int64) @ p.q.astype(np.int64), k)
            ys.append(yt)
            yds.append(mul16(yt, s))
        # token 0 is the layer's: the pack's own goldens, bit for bit
        same = (np.array_equal(bits16(ys[0]), bits16(y))
                and np.array_equal(bits16(yds[0]), bits16(yd)))
        if not same:
            raise ValueError(f"{key}: the GEMV model disagrees with the golden pack on token 0")
        a = em.blob(f"{key}_a", xs)  # the vectors alone, [NTOK, K]
        a4 = em.blob(f"{key}_a4", to_a4(xs))
        w = em.blob(f"{key}_w", p.blob(mesh))
        sb = em.blob(f"{key}_s", bits16(s))
        yb = em.blob(f"{key}_y", bits16(np.stack(ys)))
        ydb = em.blob(f"{key}_yd", bits16(np.stack(yds)))
        rows.append(f'    {{"{names[key]}", {a}, {a4}, {w}, {sb}, {yb}, {ydb}, {p.K}u, {p.N}u, '
                    f'{k}u, {p.wbits}u}},')
        if key == "router":
            neg = i
            y0 = d_port(xs.astype(np.int64) @ p.q.astype(np.int64), 0)  # [NTOK, N]
            em.blob("router_y_k0", bits16(y0))
            ninf = np.isinf(y0.astype(np.float32))
            em.c(f"// Negative control: the router at k = 0 holds {int(ninf[0].sum())} infinities"
                 f" out of {p.N} for token 0, {int(ninf.sum())} over the {NTOK} tokens.")
    em.define("GEMV_NEG_SHAPE", neg, "the router's index, or -1 when it is not built")
    em.c("typedef struct {\n    const char *name;\n    const int8_t *a, *a4, *w;\n"
         "    const uint16_t *s, *y, *yd;\n    uint32_t K, N, k;\n"
         "    uint32_t wbits;  // 8, or 4: INT4 through the B converter\n} gemv_shape_t;")
    em.c("static const gemv_shape_t gemv_shapes[GEMV_NSHAPES] = {\n" + "\n".join(rows) + "\n};")
    em.write(args.header, args.blob_dir / "blobs.S")
    sys.stderr.write(f"[dsv2-gemv datagen] {len(keys)} shapes, {em.bytes / 2**20:.1f} MiB of blobs\n")


if __name__ == "__main__":
    main()
