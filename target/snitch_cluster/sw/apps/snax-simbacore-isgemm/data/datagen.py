#!/usr/bin/env python3

# Copyright 2025 dInnerUnroll Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>

import argparse
import pathlib
import hjson
import sys
import os

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../snax-simbacore-main/data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from datagen_base import BANKWIDTH, DataGeneratorBase, BANK_BYTES, FP8, BF16  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]


class DataGenerator(DataGeneratorBase):
    APP_NAME = "isgemm"

    def __init__(self, **kwargs):
        super().__init__(self.APP_NAME, **kwargs)

    def run(self):
        self.build_isgemm_data()

    def build_isgemm_data(self):
        mode_id = 4
        assert f"M{mode_id}_ISGEMM" in self.kwargs, f"verify mode_id {mode_id} for ISGEMM"
        seqLenUnroll = self.kwargs["seqLenUnroll"]
        dInnerUnroll = self.kwargs["dInnerUnroll"]
        seqLen = self.kwargs["dim0"]
        dInner = self.kwargs["dim1"]
        dModel = self.kwargs["dim2"]
        iscore_serial_width = self.kwargs["iscore_serial_width"]

        a_in_width = seqLenUnroll * dInnerUnroll * FP8
        assert a_in_width % iscore_serial_width == 0, "a_in_width must be divisible by iscore_serial_width"

        # First inject zeros, then (K-1) times the full output matrix
        # The initial values (C) can be at the same addresses as the output matrix
        psum_bounds_and_strides = (
            [
                (seqLen // seqLenUnroll) * dModel,  # one output matrix
                dInner // dInnerUnroll,  # complete reduction dimension (K)
            ],
            [
                seqLenUnroll * BF16 // 8,
                0,  # Go to same addresses again
            ],
        )

        streamers = {
            "R11": (  # iscore in. Stored in convFormat
                [(dInner // dInnerUnroll) * (seqLen // seqLenUnroll) * (a_in_width // iscore_serial_width)],
                [iscore_serial_width // 8],
            ),
            "R12": (  # iscore weight
                [
                    dModel,  # N
                    seqLen // seqLenUnroll,  # M
                    dInner // dInnerUnroll,  # K
                ],
                [
                    dInnerUnroll * FP8 // 8,
                    0,
                    dModel * dInnerUnroll * FP8 // 8,
                ],
            ),
            "R13": psum_bounds_and_strides,
            "W3": psum_bounds_and_strides,
        }

        specs = [
            ("a", seqLen * dInner * FP8 // 8),
            ("b", dInner * dModel * FP8 // 8),
            ("cd", seqLen * dModel * BF16 // 8),  # c and d use same space
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)
        scalars = {**lengths, **deltas}

        test_data = {**{name: "uint8_t" for name in ("A", "B", "D")}, "C": "uint16_t"}
        tests = {"D": seqLen * dModel}

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
