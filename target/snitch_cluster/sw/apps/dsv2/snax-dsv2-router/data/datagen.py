#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data for snax-dsv2-router: DeepSeek-V2-Lite's MoE gate for one token, from the layer's own
# post-attention state (the golden pack, sw/apps/dsv2/util): the router GEMV (2048 -> 64) and its
# dequantisation, the softmax over the 64 logits, and the top 6. The golden token is drawn so
# the device model's sixth and seventh weights are golden.TIE_ULP or more apart: no device
# softmax within a few ULP of the model can swap them.

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))  # the util package
from util import golden  # noqa E402
from util.emit import Emitter  # noqa E402
from util.fp import bits16, ulp16  # noqa E402
from util.layout import mesh_from_hwcfg  # noqa E402


def main():
    ap = argparse.ArgumentParser(description="Data for snax-dsv2-router")
    ap.add_argument("--swcfg", type=pathlib.Path, required=True)
    ap.add_argument("--hwcfg", type=pathlib.Path, required=True)
    ap.add_argument("--header", type=pathlib.Path, required=True)
    ap.add_argument("--blob-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()
    prm = hjson.loads(args.swcfg.read_text())
    mesh = mesh_from_hwcfg(hjson.loads(args.hwcfg.read_text()))
    if mesh != (16, 4, 16):
        raise ValueError(f"the kernel's descriptors assume a (16, 4, 16) mesh, not {mesh}")

    g = golden.make(seed=int(prm["seed"]), L=int(prm["L"]))
    H, P, d = g.hw, g.pack, g.dims
    ids = np.asarray(H["ids"], dtype=np.uint32)
    order = np.argsort(-H["p16"].astype(np.float64), kind="stable")
    gap = int(ulp16(H["p16"][order[d.top_k - 1]], H["p16"][order[d.top_k]]))

    em = Emitter(args.blob_dir)
    em.c(f"// DeepSeek-V2-Lite layer 1 router, golden seed {g.seed}; the golden picks "
         f"{list(map(int, ids))}, the 6th and 7th {gap} ULP apart.")
    em.define("N_EXP", d.n_routed)
    em.define("TOP_K", d.top_k)
    em.define("K_IN", d.hidden)
    em.define("D_SHIFT", H["ks"]["x"])
    em.blob("hq_a", np.asarray(H["hq"], dtype=np.int8))  # hn quantised: the vector alone, 1 x K
    em.blob("wr", P.wr.blob(mesh))
    em.blob("wr_s", bits16(H["lg_s"]))
    em.blob("g_logits", bits16(H["logits16"]))
    em.blob("g_p", bits16(H["p16"]))
    em.blob("g_ids", ids)
    em.blob("g_w", bits16(H["w16"]))
    em.write(args.header, args.blob_dir / "blobs.S")
    sys.stderr.write(f"[dsv2-router datagen] top-6 {list(map(int, ids))}, gap {gap} ULP\n")


if __name__ == "__main__":
    main()
