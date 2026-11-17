#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# gesummv's data.h. The inputs as forge's kernels/polybench/gesummv.py draws
# them (make_inputs: alpha = 3 and beta = 2, then A and B, N rows of N, and x,
# each from rng.integers(-8, 8) at int64, so -8 to 7); out = alpha (A x) +
# beta (B x), each product and sum at acc_t (A, B and x in_t, out acc_t).
# alpha and beta are scalars, loaded at run time as axpy's alpha. A and B are
# stored row by row.

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import snitch_forge_data as sfd  # noqa: E402

ALPHA, BETA = 3, 2


def main():
    args = sfd.parse("gesummv", ["N"])
    rng, seed = sfd.generator(args.seed)
    n = args.N
    a = sfd.draw(rng, n * n, args.in_t, low=-8, high=8)
    b = sfd.draw(rng, n * n, args.in_t, low=-8, high=8)
    x = sfd.draw(rng, n, args.in_t, low=-8, high=8)
    xx = x.astype(args.acc_t)
    ax = np.sum(a.astype(args.acc_t).reshape(n, n) * xx, axis=1, dtype=args.acc_t)
    bx = np.sum(b.astype(args.acc_t).reshape(n, n) * xx, axis=1, dtype=args.acc_t)
    out = args.acc_t(ALPHA) * ax + args.acc_t(BETA) * bx
    sfd.write(
        "gesummv",
        args,
        seed,
        inputs={"A": a, "B": b, "x": x},
        outputs={"out": out},
        scalars={"alpha": ALPHA, "beta": BETA},
    )


if __name__ == "__main__":
    main()
