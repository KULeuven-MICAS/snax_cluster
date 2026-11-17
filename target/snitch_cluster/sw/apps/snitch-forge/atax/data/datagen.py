#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# atax's data.h. The inputs as forge's kernels/polybench/atax.py draws them
# (make_inputs: A, M rows of N, then x, each from rng.integers(-4, 4) at int64,
# so -4 to 3); y = A^T (A x), each product and sum at acc_t (A and x in_t, y
# acc_t), the intermediate A x at acc_t as the program keeps it. A is stored
# row by row.

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import snitch_forge_data as sfd  # noqa: E402


def main():
    args = sfd.parse("atax", ["M", "N"])
    rng, seed = sfd.generator(args.seed)
    m, n = args.M, args.N
    a = sfd.draw(rng, m * n, args.in_t, low=-4, high=4)
    x = sfd.draw(rng, n, args.in_t, low=-4, high=4)
    aa = a.astype(args.acc_t).reshape(m, n)
    t = np.sum(aa * x.astype(args.acc_t), axis=1, dtype=args.acc_t)
    y = np.sum(aa * t[:, None], axis=0, dtype=args.acc_t)
    sfd.write("atax", args, seed, inputs={"A": a, "x": x}, outputs={"y": y})


if __name__ == "__main__":
    main()
