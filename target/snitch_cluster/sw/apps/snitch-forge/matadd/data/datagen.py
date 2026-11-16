#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# matadd's data.h. The inputs as forge's kernels/elementwise/matadd.py draws them (make_inputs: A,
# then B, M rows of N each, from rng.integers(-1000, 1000) at int32, so -1000 to 999); C = A + B at
# acc_t. The matrices are stored row by row.

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import snitch_forge_data as sfd  # noqa: E402


def main():
    args = sfd.parse("matadd", ["M", "N"])
    rng, seed = sfd.generator(args.seed)
    a = sfd.draw(rng, args.M * args.N, args.in_t, dtype=np.int32)
    b = sfd.draw(rng, args.M * args.N, args.in_t, dtype=np.int32)
    c = a.astype(args.acc_t) + b.astype(args.acc_t)
    sfd.write("matadd", args, seed, inputs={"A": a, "B": b}, outputs={"C": c})


if __name__ == "__main__":
    main()
