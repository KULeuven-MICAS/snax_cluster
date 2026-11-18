#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# dma_transpose's data.h: A, M rows of N, drawn as forge's draw does (rng.integers(-1000, 1000) at
# int64, cast to in_t), and its golden A_T = A^T (N rows of M). A copy moves elements and computes
# nothing, so in_t and acc_t are one type: TYPE int8 or int64.

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import snitch_forge_data as sfd  # noqa: E402


def main():
    args = sfd.parse("dma_transpose", ["M", "N"])
    if args.in_t != args.acc_t:
        sys.exit(f"dma_transpose: TYPE={args.type} has in_t {args.in_t.__name__} and acc_t "
                 f"{args.acc_t.__name__}; a copy needs one type: give TYPE=int8 or int64")
    rng, seed = sfd.generator(args.seed)
    m, n = args.M, args.N
    a = sfd.draw(rng, m * n, args.in_t)
    a_t = a.reshape(m, n).T.reshape(-1).copy()
    sfd.write("dma_transpose", args, seed, inputs={"A": a}, outputs={"A_T": a_t})


if __name__ == "__main__":
    main()
