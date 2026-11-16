#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# matvec's data.h. The inputs as forge's kernels/blas/matvec.py draws them (make_inputs: A, M rows
# of N, then p, each from rng.integers(-1000, 1000) at int64, so -1000 to 999); q = A p, each
# product and sum at acc_t. A is stored row by row.

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import snitch_forge_data as sfd  # noqa: E402


def main():
    args = sfd.parse("matvec", ["M", "N"])
    rng, seed = sfd.generator(args.seed)
    a = sfd.draw(rng, args.M * args.N, args.in_t)
    p = sfd.draw(rng, args.N, args.in_t)
    prod = a.astype(args.acc_t).reshape(args.M, args.N) * p.astype(args.acc_t)
    q = np.sum(prod, axis=1, dtype=args.acc_t)
    sfd.write("matvec", args, seed, inputs={"A": a, "p": p}, outputs={"q": q})


if __name__ == "__main__":
    main()
