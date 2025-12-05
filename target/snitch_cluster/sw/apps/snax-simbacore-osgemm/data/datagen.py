#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
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
    APP_NAME = "osgemm"

    def __init__(self, **kwargs):
        super().__init__(self.APP_NAME, **kwargs)

    def run(self):
        self.build_osgemm_data()

    def build_osgemm_data(self):
        mode_id = 3
        assert f"M{mode_id}_OSGEMM" in self.kwargs, f"verify mode_id {mode_id} for OSGEMM"
        Mu = self.kwargs["seqLenUnroll"]
        Nu = self.kwargs["dInnerUnroll"]
        seqLen = self.kwargs["dim0"]
        dModel = self.kwargs["dim1"]
        dInner = self.kwargs["dim2"]
        oscore_serial_width = self.kwargs["oscore_serial_width"]

        # In VersaCore naming convention
        M = seqLen // Mu
        K = dModel
        N = dInner // Nu

        a_in_width = Mu * FP8
        b_in_width = Nu * FP8
        d_array_width = Mu * Nu * FP8
        assert d_array_width % oscore_serial_width == 0, "d_array_width Must be divisible by oscore_serial_width"

        streamers = {
            # for n in N (irrelevant dimension)
            #   for m in M
            #     for k in K
            #         parfor s in (tileSize / bankWidth)
            #             addr = k * tile_size + m * K * tile_size + s * bankWidth
            "R0": (  # Input A
                [K, M, N],
                [
                    a_in_width // 8,
                    K * a_in_width // 8,
                    0,
                ],
            ),
            "R1": (  # Input B
                [K, M, N],
                [
                    b_in_width // 8,
                    0,  # M is irrelevant
                    K * b_in_width // 8,
                ],
            ),
            "W0": (  # Output D
                [(d_array_width // oscore_serial_width) * M * N],
                [oscore_serial_width // 8],
            ),
        }

        specs = [
            ("a", M * K * a_in_width // 8),
            ("b", K * N * b_in_width // 8),
            # ("c", M * N * d_array_width // 8),
            ("d", M * N * d_array_width // 8),
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)
        scalars = {**lengths, **deltas}

        test_data = {name: "uint8_t" for name in ("A", "B", "D")}
        tests = {"D": seqLen * dInner}

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
