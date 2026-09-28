#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data for snax-dsv2-layer and snax-dsv2-spec: DeepSeek-V2-Lite layer 1, x -> out, pass after
# pass (golden.spec_passes): pass p runs `ntok` tokens from position starts[p]. The first token
# of pass 0 is the golden pack's; each pass reads the rows the passes before it appended, and a
# start below the end of the pass before drops its last rows (a rejected draft).
#
# IN DRAM: the model's weights (the MLA's, the router, the shared experts, and every routed
# expert some token picks), the cache copies holding the L cached rows, and per pass its
# tokens and RoPE tables (dsv2.appdata).
#
# GOLDENS, per pass and token, the device model's values: h = x + MLA(x), the top-6 ids and
# weights, and out = h + MoE(h); after the last pass, both cache copies.

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from dsv2 import appdata, golden  # noqa E402
from dsv2.emit import Emitter  # noqa E402
from dsv2.fp import bits16  # noqa E402
from dsv2.layout import key_copy, mesh_from_hwcfg, value_copy  # noqa E402


def main():
    ap = argparse.ArgumentParser(description="Data for snax-dsv2-layer / snax-dsv2-spec")
    ap.add_argument("--swcfg", type=pathlib.Path, required=True)
    ap.add_argument("--hwcfg", type=pathlib.Path, required=True)
    ap.add_argument("--header", type=pathlib.Path, required=True)
    ap.add_argument("--blob-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()
    prm = hjson.loads(args.swcfg.read_text())
    mesh = mesh_from_hwcfg(hjson.loads(args.hwcfg.read_text()))
    if mesh != (16, 4, 16):
        raise ValueError(f"the kernel's descriptors assume a (16, 4, 16) mesh, not {mesh}")
    bc, cap, ntok = int(prm["bc"]), int(prm["capacity"]), int(prm["ntok"])
    starts = [int(s) for s in prm["starts"]]

    g = golden.make(seed=int(prm["seed"]), L=int(prm["L"]), bc=bc)
    d = g.dims
    if starts[0] != g.L:
        raise ValueError(f"the first pass starts at the golden token's position {g.L}")
    passes = golden.spec_passes(g, starts, ntok)
    rows = np.concatenate([passes[-1]["c8"], passes[-1]["kpe8"]], axis=1)
    if cap % bc or cap < max(s + ntok for s in starts):
        raise ValueError(f"capacity {cap} must be whole tiles of {bc} and hold every pass's rows")
    ids = sorted({e for q in passes for e in q["J"]["order"]})

    em = Emitter(args.blob_dir)
    em.c(f"// DeepSeek-V2-Lite layer 1, golden seed {g.seed}: {len(passes)} passes of {ntok} "
         f"token(s) from positions {starts}; experts {ids} are in the ELF.")
    appdata.model_defines(em, g, bc, cap)
    appdata.mla_weights(em, g, mesh)
    appdata.cache(em, g, cap, mesh)
    appdata.mla_passes(em, passes, d.heads)
    appdata.moe_weights(em, g, ids, mesh)

    golds = []
    for p, q in enumerate(passes):
        toks = q["toks"]
        pad8 = [np.pad(np.asarray(tk["ids"], dtype=np.uint32), (0, 8 - d.top_k)) for tk in toks]
        em.blob(f"g_h_{p}", bits16(np.concatenate([tk["h16"] for tk in toks])))
        em.blob(f"g_ids_{p}", np.concatenate(pad8))
        em.blob(f"g_w_{p}", bits16(np.concatenate([np.pad(tk["w16"], (0, 8 - d.top_k))
                                                   for tk in toks])))
        em.blob(f"g_out_{p}", bits16(np.concatenate([tk["out16"] for tk in toks])))
        golds.append(f"{{g_h_{p}, g_ids_{p}, g_w_{p}, g_out_{p}}}")
    em.define("LAYER_GOLDENS", "{" + ", ".join(golds) + "}")
    em.define("END_ROWS", rows.shape[0], "cache rows after the last pass")
    em.blob("g_key", key_copy(rows, cap, mesh))
    em.blob("g_val", value_copy(rows[:, :d.kv_rank], cap, mesh))
    em.write(args.header, args.blob_dir / "blobs.S")
    sys.stderr.write(f"[dsv2-layer datagen] {len(passes)} passes x {ntok} token(s) from {starts}, "
                     f"experts {ids}; {em.bytes / 2**20:.1f} MiB of blobs\n")


if __name__ == "__main__":
    main()
