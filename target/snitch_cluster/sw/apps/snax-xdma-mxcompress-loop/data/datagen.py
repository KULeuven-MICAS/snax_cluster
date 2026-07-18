#!/usr/bin/env python3
# Copyright 2026 KU Leuven. SPDX-License-Identifier: Apache-2.0
# BF16 input for the in-transit MX compress -> decompress loopback. Values are E2M1-representable
# (base {2,3,4,6} x 2^k) so the BF16 -> MXFP4 -> BF16 round-trip is EXACT (recovery check).
import argparse
import hjson
import numpy as np


def bf16(f):
    return int((np.float32(f).view(np.uint32) >> 16) & 0xFFFF)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--cfg", required=True)
    a = p.parse_args()
    N = hjson.load(open(a.cfg))["N"]
    rng = np.random.default_rng(7)
    # WIDE dynamic range (powers of two {2,4,8,16}, up to 8x within a 32-elem block): with block max 16 the E8M0
    # scale is 4, so the min element 2 maps to 0.5 -- the E2M1 SUBNORMAL. This is the case a real MXFP4 block hits
    # and the one the narrowFin subnormal-encode fix restores; {0.5,1,2,4} are all exact E2M1 codes so recovery
    # is bit-exact. (The earlier {2,3,4,6}x2^k set had non-representable values AND masked the subnormal FTZ bug.)
    vals = (2.0 ** rng.integers(1, 5, N)).astype(np.float32)  # {2,4,8,16}
    bits = [bf16(v) for v in vals]
    print("#include <stdint.h>")
    print(f"#define N {N}")
    print("uint16_t bf16_input[N] __attribute__((aligned(64))) = {" + ",".join(map(str, bits)) + "};")


if __name__ == "__main__":
    main()
