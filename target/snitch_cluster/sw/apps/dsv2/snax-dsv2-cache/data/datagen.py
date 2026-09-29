#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data for snax-dsv2-cache: DeepSeek-V2-Lite's latent cache, two copies in DRAM (layout.py
# key_copy / value_copy), holding the golden pack's L cached rows, and the rows three
# passes append at positions L, L+1 and L+2:
#   L      the golden token's own row (the device model's c8 and kpe8)
#   L+1..  rows of new tokens, through the same device path (hwmodel.kv_row)
# and both copies as they must look after the three appends.

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))  # the util package
from util import golden, hwmodel  # noqa E402
from util.emit import Emitter  # noqa E402
from util.fp import F16  # noqa E402
from util.layout import key_copy, mesh_from_hwcfg, value_copy  # noqa E402


def main():
    ap = argparse.ArgumentParser(description="Data for snax-dsv2-cache")
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
        raise ValueError(f"the cache layout assumes a (16, 4, 16) mesh, not {mesh}")
    cap, n_app = int(prm["capacity"]), int(prm["appends"])

    g = golden.make(seed=int(prm["seed"]), L=int(prm["L"]), wbits=args.wbits)
    L, d = g.L, g.dims
    if cap % 16 or L + n_app > cap:
        raise ValueError(f"capacity {cap} must be a multiple of 16 holding {L + n_app} tokens")
    rows0 = np.concatenate([g.c8, g.kpe8], axis=1)                       # [L, 576]
    rng = np.random.default_rng([g.seed, 11])
    new = [np.concatenate([g.hw["c8_new"], g.hw["kpe8_new"]])]
    for i in range(1, n_app):
        x = rng.normal(0.0, 1.0, size=d.hidden).astype(F16)
        new.append(hwmodel.kv_row(g.pack, g.scales, x, L + i))
    new = np.stack(new)
    rows = np.concatenate([rows0, new])

    em = Emitter(args.blob_dir)
    em.define("DSV2_DATA_WBITS", g.wbits, "the weights' width; the kernel's DSV2_WBITS must match")
    em.c(f"// DeepSeek-V2-Lite layer 1 latent cache, golden seed {g.seed}: {L} rows cached.")
    em.define("CACHE_CAP", cap, "tokens the copies hold")
    em.define("CACHE_L", L, "rows cached before the appends")
    em.define("N_APPEND", n_app, "appends, at positions L, L+1, ...")
    em.blob("key", key_copy(rows0, cap, mesh), writable=True)
    em.blob("val", value_copy(rows0[:, :d.kv_rank], cap, mesh), writable=True)
    em.blob("new_rows", new)
    em.blob("g_key", key_copy(rows, cap, mesh))
    em.blob("g_val", value_copy(rows[:, :d.kv_rank], cap, mesh))
    em.write(args.header, args.blob_dir / "blobs.S")
    sys.stderr.write(f"[dsv2-cache datagen] {em.bytes / 2**10:.0f} KiB of blobs\n")


if __name__ == "__main__":
    main()
