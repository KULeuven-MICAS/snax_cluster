#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# axpy's data.h. The inputs as forge's kernels/blas/axpy.py draws them (make_inputs: alpha = 3,
# then x, then y, each from rng.integers(-100, 100) at int32, so -100 to 99); y += alpha * x in
# place, at acc_t (forge's presets give y and alpha acc_t, x in_t).

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import snitch_forge_data as sfd  # noqa: E402

ALPHA = 3


def main():
    args = sfd.parse("axpy", ["N"])
    rng, seed = sfd.generator(args.seed)
    x = sfd.draw(rng, args.N, args.in_t, low=-100, high=100, dtype=np.int32)
    y = sfd.draw(rng, args.N, args.acc_t, low=-100, high=100, dtype=np.int32)
    out = y + args.acc_t(ALPHA) * x.astype(args.acc_t)
    sfd.write(
        "axpy",
        args,
        seed,
        inputs={"x": x},
        outputs={},
        inouts={"y": (y, out)},
        scalars={"alpha": ALPHA},
    )


if __name__ == "__main__":
    main()
