#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# rowsum's data.h. The input as forge's kernels/elementwise/rowsum.py draws it (make_inputs: A, M
# rows of N, from rng.integers(-100, 100) at int64, so -100 to 99); y = the sum of each row of A at
# acc_t. A is stored row by row.

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import snitch_forge_data as sfd  # noqa: E402


def main():
    args = sfd.parse("rowsum", ["M", "N"])
    rng, seed = sfd.generator(args.seed)
    a = sfd.draw(rng, args.M * args.N, args.in_t, low=-100, high=100)
    y = np.sum(a.astype(args.acc_t).reshape(args.M, args.N), axis=1, dtype=args.acc_t)
    sfd.write("rowsum", args, seed, inputs={"A": a}, outputs={"y": y})


if __name__ == "__main__":
    main()
