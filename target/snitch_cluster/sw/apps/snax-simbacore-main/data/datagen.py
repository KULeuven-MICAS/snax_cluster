#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>

import pathlib
import sys
import os

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from datagen_base import BANKWIDTH, DataGeneratorBase, BANK_BYTES, FP8, BF16  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]


class DataGenerator(DataGeneratorBase):
    APP_NAME = "main"

    def __init__(self, **kwargs):
        super().__init__(self.APP_NAME, **kwargs)

    def run(self):
        self.save_params()
        self.build_Phase1_data()
        self.build_Phase2_data()

    def save_params(self):
        """Saves params to self as shorthand notation"""
        # Algorithm
        self.seqLen = self.kwargs["seqLen"]
        self.dModel = self.kwargs["dModel"]
        self.dInner = self.kwargs["dInner"]
        self.dtRank = self.kwargs["dtRank"]
        self.dConv = self.kwargs["dConv"]
        self.dState = self.kwargs["dState"]
        self.xProjDim = self.kwargs["xProjDim"]
        # HW dimensions
        self.seqLenUnroll = self.kwargs["seqLenUnroll"]
        self.dInnerUnroll = self.kwargs["dInnerUnroll"]
        self.dtRankUnroll = self.kwargs["dtRankUnroll"]
        self.convUnroll = self.kwargs["convUnroll"]
        self.delaySU = self.kwargs["delaySU"]
        # HW widths
        self.oscore_serial_width = self.kwargs["oscore_serial_width"]
        self.switchcore_width = self.kwargs["switchcore_width"]
        self.iscore_serial_width = self.kwargs["iscore_serial_width"]
        self.suc_serial_width_A = self.kwargs["suc_serial_width_A"]
        self.suc_serial_width_BC = self.kwargs["suc_serial_width_BC"]  # Streamer width is 2x this value!
        self.switchcore_width = self.kwargs["switchcore_width"]

    def get_safe_to_start_delay(self):
        """In Phase2, the SU core reads the OS core output from memory, in a different order. The program must ensure
        that when the SU core streamer starts, all memory contents will be valid by the time they are read. The
        safe-to-start time depends on (self.seqLen, self.dModel, self.dInner), as the relative throughput of the OS and SU cores
        changes. The same is true for the SU core output to IS core input.

        This function returns after how many OS core tiles the SU core can start, and after SU core output elements the
        IS core can start. Both values can be compared to the CSR registers.
        The safe-to-start time is computed as: (time to complete one window) * max(throughput ratio, 1)
        """
        # OS core and IS core have same throughput
        gemm_total_nb_tiles = (self.seqLen // self.seqLenUnroll) * (self.dInner // self.dInnerUnroll)
        suc_total_nb_elements = self.seqLen * self.dInner
        gemm_tp = 1 / (gemm_total_nb_tiles * self.dModel)  # dModel / tile
        suc_tp = 1 / suc_total_nb_elements  # 1 cc / elem

        gemm_window_cnt = self.seqLen // self.seqLenUnroll  # expressed in OS core tiles
        suc_window_cnt = self.seqLen * self.dInnerUnroll  # expressed in SUC output elements

        suc_safe_to_start = gemm_window_cnt * max(suc_tp / gemm_tp, 1)  # expressed in OS core tiles
        iscore_safe_to_start = suc_window_cnt * max(gemm_tp / suc_tp, 1)  # expressed in SUC output elements

        # Make sure the delay does not exceed the total number of tiles or elements
        return int(min(suc_safe_to_start, gemm_total_nb_tiles)), int(min(iscore_safe_to_start, suc_total_nb_elements))

    def build_Phase1_data(self):
        mode_id = 1
        assert f"M{mode_id}_PHASE1" in self.kwargs, "verify mode_id"
        assert self.switchcore_width == BANKWIDTH

        streamers = {
            "R0": (  # osCore in
                [
                    self.dModel,  # K
                    self.seqLen // self.seqLenUnroll,  # M
                    self.dInner // self.dInnerUnroll,  # N
                ],
                [
                    self.seqLenUnroll * FP8 // 8,
                    self.dModel * self.seqLenUnroll * FP8 // 8,
                    0,
                ],
            ),
            "R1": (  # oscore weight
                [
                    self.dModel,  # K
                    self.seqLen // self.seqLenUnroll,  # M
                    self.dInner // self.dInnerUnroll,  # N
                ],
                [
                    self.dInnerUnroll * FP8 // 8,
                    0,
                    self.dModel * self.dInnerUnroll * FP8 // 8,
                ],
            ),
            "R3": (  #  conv (switchCore) weight: layout is row-major [dInner, dConv]
                [self.dConv * self.dInner * FP8 // BANKWIDTH],
                [BANK_BYTES],
            ),
            "R4": (  #  conv (switchCore) bias: layout is row-major [dInner]
                [self.dInner * FP8 // BANKWIDTH],
                [BANK_BYTES],
            ),
            "R12": (  # iscore weight
                [
                    self.xProjDim,  # N
                    self.seqLen // self.seqLenUnroll,  # M
                    self.dInner // self.dInnerUnroll,  # K
                ],
                [
                    self.dInnerUnroll * FP8 // 8,
                    0,
                    self.xProjDim * self.dInnerUnroll * FP8 // 8,
                ],
            ),
            "R13": (  # isCore psum
                # First inject zeros, then (K-1) times the full output matrix
                # The initial values (C) can be at the same addresses as the output matrix
                [
                    (self.seqLen // self.seqLenUnroll) * self.xProjDim,  # one output matrix
                    self.dInner // self.dInnerUnroll,  # complete reduction dimension (K)
                ],
                [
                    self.seqLenUnroll * BF16 // 8,
                    0,  # Go to same addresses again
                ],
            ),
            "W1": (  # conv output
                [self.seqLen * self.dInner * FP8 // BANKWIDTH],
                [BANK_BYTES],
            ),
            "W3": (  # isCore output: EXACTLY the same as psum reader R13
                [
                    (self.seqLen // self.seqLenUnroll) * self.xProjDim,
                    self.dInner // self.dInnerUnroll,
                ],
                [
                    self.seqLenUnroll * BF16 // 8,
                    0,
                ],
            ),
        }

        specs = [
            ("oscore_in", self.seqLen * self.dModel * FP8 // 8),
            ("oscore_weight", self.dModel * self.dInner * FP8 // 8),
            ("conv_weight", self.dInner * self.dConv * FP8 // 8),
            ("conv_bias", self.dInner * FP8 // 8),
            ("conv_out", self.seqLen * self.dInner * FP8 // 8),
            ("iscore_weight", self.dInner * self.xProjDim * FP8 // 8),
            # Reserve space for BF16 psums. On the final iteration, only first half will contain valid data
            ("iscore_out", self.seqLen * self.xProjDim * BF16 // 8),
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)
        scalars = {**lengths, **deltas}

        # Sampled outputs plus full tensor payloads.
        tests = {"conv_out": self.seqLen * self.dInner, "iscore_out": self.seqLen * self.xProjDim}

        test_data = {
            name: "uint8_t"
            for name in (
                "oscore_in",
                "oscore_weight",
                "conv_weight",
                "conv_bias",
                "conv_out",
                "iscore_weight",
                "iscore_bias",
                "iscore_out",
            )
        }

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)

    def build_Phase2_data(self):
        mode_id = 2
        assert f"M{mode_id}_PHASE2" in self.kwargs, "verify mode_id"

        # assert self.convUnroll * FP8 == BANK_BYTES * 8, "switchCore output width must match 1 bank width"
        # assert self.dConv * FP8 == BANK_BYTES * 8, "switchCore weight width must match 1 bank width"
        assert self.dtRank * FP8 % self.switchcore_width == 0, "dtRank must be divisible by switchCore elem/cc in"
        assert self.switchcore_width == BANKWIDTH, "switchcore_width must match bank width"

        suc_parallel_widthA = self.dState * FP8
        # suc_parallel_widthBC = self.dState * FP8
        switchcore_parallel_width_W1 = self.convUnroll * self.dConv * FP8
        switchcore_parallel_width_W2 = self.convUnroll * (self.dtRankUnroll - self.dConv) * FP8
        # switchcore_parallel_width_bias = self.convUnroll * FP8
        oscore_parallel_width_d = self.seqLenUnroll * self.dInnerUnroll * FP8
        iscore_parallel_width_d = self.seqLenUnroll * self.dInnerUnroll * FP8

        # Reads out a layout that is stored in convFormat, in SUC format ordering
        bounds_conv_to_suc = [
            (self.convUnroll * self.seqLenUnroll) // (BANKWIDTH // FP8),  # subTileSize / (elem per transfer)
            self.seqLen // self.seqLenUnroll,  # tiles per window
            self.dInnerUnroll // self.convUnroll,  # subtiles per tile
            self.dInner // self.dInnerUnroll,  # windows per tensor
        ]
        strides_conv_to_suc = [
            BANK_BYTES,
            self.seqLenUnroll * self.dInnerUnroll * FP8 // 8,  # tile size
            self.convUnroll * self.seqLenUnroll * FP8 // 8,  # subtile size
            self.seqLen * self.dInnerUnroll * FP8 // 8,  # window size
        ]

        streamers = {
            "R0": (  # osCore in
                [
                    self.dModel,  # K
                    self.seqLen // self.seqLenUnroll,  # M
                    self.dInner // self.dInnerUnroll,  # N
                ],
                [
                    self.seqLenUnroll * FP8 // 8,
                    self.dModel * self.seqLenUnroll * FP8 // 8,
                    0,
                ],
            ),
            "R1": (  # oscore weight
                [
                    self.dModel,  # K
                    self.seqLen // self.seqLenUnroll,  # M
                    self.dInner // self.dInnerUnroll,  # N
                ],
                [
                    self.dInnerUnroll * FP8 // 8,
                    0,
                    self.dModel * self.dInnerUnroll * FP8 // 8,
                ],
            ),
            "R2": (  # switchCore in (deltaMinor)
                [
                    self.dtRank * FP8 // self.switchcore_width,  # K
                    self.seqLenUnroll,
                    self.seqLen // self.seqLenUnroll,
                    self.dInner // self.convUnroll,  # N (irrelevant dimension)
                ],
                [
                    (self.switchcore_width // 8) * self.seqLenUnroll,
                    BANK_BYTES,
                    self.seqLenUnroll * self.xProjDim * FP8 // 8,
                    0,
                ],
                self.seqLenUnroll * BANK_BYTES,  # Spatial stride
            ),
            "R3": (  # switchCore weight (partition 1). Weights rotate internally so no reuse in self.seqLen here
                [
                    (self.dtRank // self.dtRankUnroll)
                    * (self.dInner // self.convUnroll)
                    * (switchcore_parallel_width_W1 // self.switchcore_width)  # serDes factor
                ],
                [self.switchcore_width // 8],
            ),
            "R4": (  #  switchCore bias
                [(self.dInner * FP8) // self.switchcore_width],
                [self.switchcore_width // 8],
            ),
            "R5": (  #  switchCore weight (partition 2)
                [
                    (self.dtRank // self.dtRankUnroll)
                    * (self.dInner // self.convUnroll)
                    * (switchcore_parallel_width_W2 // self.switchcore_width)  # serDes factor
                ],
                [self.switchcore_width // 8],
            ),
            "R6": (  # SUC A
                [self.dInner * (suc_parallel_widthA // self.suc_serial_width_A)],
                [self.suc_serial_width_A // 8],
            ),
            "R7": (  # SUC BC. Packed together with dt
                [
                    (2 * self.dState * FP8) // (2 * self.suc_serial_width_BC),  #
                    self.seqLenUnroll,
                    self.seqLen // self.seqLenUnroll,
                    self.dInner // self.delaySU,  # Irrelevant dimension
                ],
                [
                    (2 * self.suc_serial_width_BC // 8) * self.seqLenUnroll,
                    BANK_BYTES,
                    self.seqLenUnroll * self.xProjDim * FP8 // 8,
                    0,
                ],
                self.seqLenUnroll * BANK_BYTES,  # Spatial stride
            ),
            "R8": (  # SUC D
                [self.dInner * FP8 // BANKWIDTH],
                [BANK_BYTES],
            ),
            "R9": (bounds_conv_to_suc, strides_conv_to_suc),  # SUC x
            "R10": (bounds_conv_to_suc, strides_conv_to_suc),  # SUC z
            "R11": (  # iscore in. Stored in convFormat
                [
                    (self.dInner // self.dInnerUnroll)
                    * (self.seqLen // self.seqLenUnroll)
                    * (iscore_parallel_width_d // self.iscore_serial_width)
                ],
                [self.iscore_serial_width // 8],
            ),
            "R12": (  # iscore weight
                [
                    self.dModel,  # N
                    self.seqLen // self.seqLenUnroll,  # M
                    self.dInner // self.dInnerUnroll,  # K
                ],
                [
                    self.dInnerUnroll * FP8 // 8,
                    0,
                    self.dModel * self.dInnerUnroll * FP8 // 8,
                ],
            ),
            "R13": (  # isCore psum
                # First inject zeros, then (K-1) times the full output matrix
                # The initial values (C) can be at the same addresses as the output matrix
                [
                    (self.seqLen // self.seqLenUnroll) * self.dModel,  # one output matrix
                    self.dInner // self.dInnerUnroll,  # complete reduction dimension (K)
                ],
                [
                    self.seqLenUnroll * BF16 // 8,
                    0,  # Go to same addresses again
                ],
            ),
            "W0": (  # osCore out: writes in convFormat
                [
                    (oscore_parallel_width_d // self.oscore_serial_width)
                    * (self.seqLen // self.seqLenUnroll)
                    * (self.dInner // self.dInnerUnroll)
                ],
                [self.oscore_serial_width // 8],
            ),
            "W2": (  # SUC output y. Produced in SUC format, must be stored in convFormat
                bounds_conv_to_suc,
                strides_conv_to_suc,
            ),
            "W3": (  # isCore output: EXACTLY the same as psum reader R13
                [
                    (self.seqLen // self.seqLenUnroll) * self.dModel,
                    self.dInner // self.dInnerUnroll,
                ],
                [
                    self.seqLenUnroll * BF16 // 8,
                    0,
                ],
            ),
        }

        tensor_size = self.seqLen * self.dInner * FP8 // 8
        specs = [
            ("oscore_in", self.seqLen * self.dModel * FP8 // 8),
            ("oscore_weight", self.dModel * self.dInner * FP8 // 8),
            ("z", tensor_size),
            # ("dt_in", self.dInner * self.dtRank * FP8 // 8),
            ("dt_BC", self.seqLen * self.xProjDim * FP8 // 8),
            ("dt_weight_1", self.dInner * (self.dtRank // self.dtRankUnroll) * self.dConv * FP8 // 8),
            (
                "dt_weight_2",
                self.dInner * (self.dtRank // self.dtRankUnroll) * (self.dtRankUnroll - self.dConv) * FP8 // 8,
            ),
            ("dt_bias", self.dInner * FP8 // 8),
            ("x", tensor_size),
            ("A", self.dInner * self.dState * FP8 // 8),
            # ("BC", 2 * self.seqLen * self.dState * FP8 // 8),
            ("D", self.dInner * FP8 // 8),
            ("y", tensor_size),
            ("iscore_weight", self.dModel * self.dInner * FP8 // 8),
            ("iscore_out", self.seqLen * self.dModel * BF16 // 8),
        ]

        lengths, deltas = self._collect_lengths_and_deltas(specs)
        suc_start_cnt, iscore_start_cnt = self.get_safe_to_start_delay()
        scalars = {
            **lengths,
            **deltas,
            "R10_start_cnt": suc_start_cnt,  # R10 is SUC input z: comes from OS core
            "R11_start_cnt": iscore_start_cnt,  # R11 is IS core input, comes from SUC output y
            "dt_to_BC_offset": self.seqLenUnroll * self.dtRank * FP8 // 8,  # First BC value in dt_BC tensor
        }

        tests = {
            "z": self.seqLen * self.dInner,
            "y": self.seqLen * self.dInner,
            "iscore_out": self.seqLen * self.dModel,
        }

        test_data = {
            name: "uint8_t"
            for name in (
                "oscore_in",
                "oscore_weight",
                "oscore_expected",  # aka matrix z
                "dt_BC",
                "dt_weight_1",
                "dt_weight_2",
                "dt_bias",
                # "suc_state",
                "suc_A",
                "suc_D",
                "suc_x",
                "suc_expected",
                "iscore_weight",
                "iscore_bias",
                "iscore_expected",
            )
        }

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests, app_name=self.APP_NAME)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
