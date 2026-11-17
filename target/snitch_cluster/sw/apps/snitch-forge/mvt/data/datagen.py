#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# mvt's data.h. The inputs as forge's kernels/polybench/mvt.py draws them (make_inputs: A, M rows
# of N, then x1, x2, y1, y2, each from rng.integers(-8, 8) at int64, so -8 to 7); x1 += A y1 and
# x2 += A^T y2 in place, each product and sum at acc_t (A and the y in_t, the x acc_t, as forge's
# presets give them). A is stored row by row; y1 is y1_l2 in data.h, as in forge's, since y1 is
# also libm's Bessel function.

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import snitch_forge_data as sfd  # noqa: E402


def main():
    args = sfd.parse("mvt", ["M", "N"])
    rng, seed = sfd.generator(args.seed)
    m, n = args.M, args.N
    a = sfd.draw(rng, m * n, args.in_t, low=-8, high=8)
    x1 = sfd.draw(rng, m, args.acc_t, low=-8, high=8)
    x2 = sfd.draw(rng, n, args.acc_t, low=-8, high=8)
    y1 = sfd.draw(rng, n, args.in_t, low=-8, high=8)
    y2 = sfd.draw(rng, m, args.in_t, low=-8, high=8)
    aa = a.astype(args.acc_t).reshape(m, n)
    x1_out = x1 + np.sum(aa * y1.astype(args.acc_t), axis=1, dtype=args.acc_t)
    x2_out = x2 + np.sum(aa * y2.astype(args.acc_t)[:, None], axis=0, dtype=args.acc_t)
    sfd.write(
        "mvt",
        args,
        seed,
        inputs={"A": a, "y1_l2": y1, "y2": y2},
        outputs={},
        inouts={"x1": (x1, x1_out), "x2": (x2, x2_out)},
    )


if __name__ == "__main__":
    main()
