#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# relu's data.h. The input as forge's kernels/elementwise/relu.py draws it (make_inputs: A from
# rng.integers(-1000, 1000) at int32, so -1000 to 999); B = max(A, 0) at acc_t.

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import snitch_forge_data as sfd  # noqa: E402


def main():
    args = sfd.parse("relu", ["N"])
    rng, seed = sfd.generator(args.seed)
    a = sfd.draw(rng, args.N, args.in_t, dtype=np.int32)
    b = np.maximum(a.astype(args.acc_t), 0)
    sfd.write("relu", args, seed, inputs={"A": a}, outputs={"B": b})


if __name__ == "__main__":
    main()
