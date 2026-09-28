#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data for snax-dsv2-moe: DeepSeek-V2-Lite layer 1's mixture of experts for one token,
# h -> h + MoE(h) (stages 14 to 23), from the golden pack (sw/apps/dsv2/util).
#
# THE EXPERT TABLE (dsv2.appdata.moe_weights). 64 entries, one per routed expert: its fused
# gate|up weight [2048, 2816], its down weight [1408, 2048] (both B-layout blobs), the
# per-column dequantisation factors of both GEMVs (FP16, with the static activation scales
# folded in), and its SwiGLU output scale. Only the experts the golden token picks have
# weights in the ELF; every other entry is zero, and the kernel refuses a zero entry rather
# than read address 0. The shared experts (fused hidden size 2816) have their own entry.
#
# GOLDENS are the device model's chained values, bit-exact at every stage (the model
# reproduces the SIMD's tables and orders of summation): hn, its INT8 values (the A operand's
# row 0), the logits, p, the
# ids and weights, per slot the dequantised gate|up and the down output, the shared ones, and
# the output.

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))  # the util package
from util import appdata, golden  # noqa E402
from util.emit import Emitter  # noqa E402
from util.fp import bits16  # noqa E402
from util.layout import mesh_from_hwcfg  # noqa E402


def main():
    ap = argparse.ArgumentParser(description="Data for snax-dsv2-moe")
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
    H = g.hw
    ids = [int(e) for e in H["ids"]]

    em = Emitter(args.blob_dir)
    em.c(f"// DeepSeek-V2-Lite layer 1 MoE, golden seed {g.seed}: experts {ids} are in the ELF.")
    appdata.model_defines(em, g, g.bc, g.L + 1)
    appdata.moe_weights(em, g, ids, mesh)
    em.blob("h16", bits16(H["h16"]))
    em.blob("g_hn", bits16(H["hn"]))
    em.blob("g_ha", np.asarray(H["hq"], dtype=np.int8))  # the A operand's row 0
    em.blob("g_logits", bits16(H["logits16"]))
    em.blob("g_p", bits16(H["p16"]))
    em.blob("g_ids", np.asarray(ids, dtype=np.uint32))
    em.blob("g_w", bits16(H["w16"]))
    em.blob("g_gu", bits16(np.stack([s["g16"] for s in H["slots"]])).reshape(-1))
    em.blob("g_y", bits16(np.stack([s["y16"] for s in H["slots"]])).reshape(-1))
    em.blob("g_sh_gu", bits16(H["shared"]["g16"]))
    em.blob("g_sh_y", bits16(H["shared"]["y16"]))
    em.blob("g_out", bits16(H["out16"]))
    em.write(args.header, args.blob_dir / "blobs.S")
    sys.stderr.write(f"[dsv2-moe datagen] experts {ids}; {em.bytes / 2**20:.1f} MiB of blobs\n")


if __name__ == "__main__":
    main()
