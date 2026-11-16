#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# vsum's data.h. The input as forge's kernels/polybench/vsum.py draws it (make_inputs: A from
# rng.integers(-1000, 1000), so -1000 to 999); out = the sum of A at acc_t.

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import snitch_forge_data as sfd  # noqa: E402


def main():
    args = sfd.parse("vsum", ["N"])
    rng, seed = sfd.generator(args.seed)
    a = sfd.draw(rng, args.N, args.in_t)
    out = np.array([np.sum(a.astype(args.acc_t), dtype=args.acc_t)], dtype=args.acc_t)
    sfd.write("vsum", args, seed, inputs={"A": a}, outputs={"out": out})


if __name__ == "__main__":
    main()
