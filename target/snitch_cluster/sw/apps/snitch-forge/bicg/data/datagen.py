#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# bicg's data.h. The inputs as forge's kernels/polybench/bicg.py draws them
# (make_inputs: A, M rows of N, then p and r, each from rng.integers(-8, 8) at
# int64, so -8 to 7); q = A p and s = A^T r, each product and sum at acc_t (A,
# p and r in_t, q and s acc_t). A is stored row by row.

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import snitch_forge_data as sfd  # noqa: E402


def main():
    args = sfd.parse("bicg", ["M", "N"])
    rng, seed = sfd.generator(args.seed)
    m, n = args.M, args.N
    a = sfd.draw(rng, m * n, args.in_t, low=-8, high=8)
    p = sfd.draw(rng, n, args.in_t, low=-8, high=8)
    r = sfd.draw(rng, m, args.in_t, low=-8, high=8)
    aa = a.astype(args.acc_t).reshape(m, n)
    q = np.sum(aa * p.astype(args.acc_t), axis=1, dtype=args.acc_t)
    s = np.sum(aa * r.astype(args.acc_t)[:, None], axis=0, dtype=args.acc_t)
    sfd.write("bicg", args, seed, inputs={"A": a, "p": p, "r": r}, outputs={"q": q, "s": s})


if __name__ == "__main__":
    main()
