#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# vecadd's data.h. The inputs as forge's kernels/polybench/vecadd.py draws them (make_inputs: A,
# then B, each from rng.integers(-1000, 1000), so -1000 to 999); C = A + B at acc_t.

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import snitch_forge_data as sfd  # noqa: E402


def main():
    args = sfd.parse("vecadd", ["N"])
    rng, seed = sfd.generator(args.seed)
    a = sfd.draw(rng, args.N, args.in_t)
    b = sfd.draw(rng, args.N, args.in_t)
    c = a.astype(args.acc_t) + b.astype(args.acc_t)
    sfd.write("vecadd", args, seed, inputs={"A": a, "B": b}, outputs={"C": c})


if __name__ == "__main__":
    main()
