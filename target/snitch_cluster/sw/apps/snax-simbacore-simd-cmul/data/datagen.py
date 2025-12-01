#!/usr/bin/env python3

# Copyright 2025 dInnerUnroll Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>

import pathlib
import sys
import os

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../snax-simbacore-main/data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from datagen_base import BANKWIDTH, DataGeneratorBase, BANK_BYTES, FP8, BF16  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]


class DataGenerator(DataGeneratorBase):

    def run(self):
        self.build_SIMD_CMUL()

    def build_SIMD_CMUL(self):
        mode_id = 5
        assert f"M{mode_id}_SIMD_CMUL" in self.kwargs, "verify mode_id"

        seqLen = self.kwargs["seqLen"]
        width = 2 * self.kwargs["suc_serial_width_BC"]

        # This must be equal to the other ports as well. Can we assert this?
        assert width == BF16 * self.kwargs["seqLenUnroll"]

        # We defined L/2 complex values in the scala generator
        bounds_and_strides = ([seqLen * BF16 // width], [width // 8])

        streamers = {
            "R7": bounds_and_strides,  # Input A (real and imag are interleaved)
            "R13": bounds_and_strides,  # Input B
            "W3": bounds_and_strides,  # Output
        }

        specs = [(tensor_name, seqLen * BF16 // 8) for tensor_name in ("cmul_a", "cmul_b", "cmul_out")]

        lengths, deltas = self._collect_lengths_and_deltas(specs)
        scalars = {**lengths, **deltas}
        tests = {"cmul_out": seqLen}
        test_data = {name: "uint16_t" for name in ("cmul_a", "cmul_b", "cmul_out")}

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
