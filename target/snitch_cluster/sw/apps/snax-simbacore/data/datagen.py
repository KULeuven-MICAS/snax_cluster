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
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

# try:
#     from .datagen_util import DataGeneratorBase, BANKWIDTH  # type: ignore[import]
# except ImportError:  # pragma: no cover - executed when run as a script
from datagen_util import BANKWIDTH, DataGeneratorBase, BANK_BYTES  # type: ignore[import]


NBIT = 16  # BF16


class DataGenerator(DataGeneratorBase):

    def run(self):
        self.format_params()
        self.build_OSGeMM_data()
        self.build_Phase1_data()

    def build_OSGeMM_data(self):
        mode = self.kwargs["M2_OSGEMM"]
        seqLen = self.kwargs["seqLen"]
        dModel = self.kwargs["dModel"]
        dInner = self.kwargs["dInner"]
        Mu = self.kwargs["seqLenUnroll"]
        Nu = self.kwargs["dInnerUnroll"]
        serial_width_d = self.kwargs["serial_width_d"]

        # In VersaCore naming convention
        M = seqLen // Mu
        K = dModel
        N = dInner // Nu

        a_array_width = Mu * NBIT
        b_array_width = Nu * NBIT
        d_array_width = Mu * Nu * NBIT
        assert d_array_width % serial_width_d == 0, "d_array_width Must be divisible by serial_width_d"

        streamers = {
            # for n in N (irrelevant dimension)
            #   for m in M
            #     for k in K
            #         parfor s in (tileSize / bankWidth)
            #             addr = k * tile_size + m * K * tile_size + s * bankWidth
            "R0": (  # Input A
                [K, M, N],
                [
                    a_array_width // 8,
                    K * a_array_width // 8,
                    0,
                ],
            ),
            "R1": (  # Input B
                [K, M, N],
                [
                    b_array_width // 8,
                    0,  # M is irrelevant
                    K * b_array_width // 8,
                ],
            ),
            "W0": (  # Output D
                [(d_array_width // serial_width_d) * M * N],
                [serial_width_d // 8],
            ),
        }

        specs = [
            ("a", M * K * a_array_width // 8),
            ("b", K * N * b_array_width // 8),
            # ("c", M * N * d_array_width // 8),
            ("d", M * N * d_array_width // 8),
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)
        scalars = {**lengths, **deltas}

        test_data = {name: "uint16_t" for name in ("A", "B", "D")}
        tests = {"D": seqLen * dInner}

        self.build_mode(mode, streamers, scalars=scalars, test_data=test_data, tests=tests)

    def build_Phase1_data(self):
        mode = self.kwargs["M0_PHASE1"]
        seqLen = self.kwargs["seqLen"]
        dModel = self.kwargs["dModel"]
        dInner = self.kwargs["dInner"]
        seqLenUnroll = self.kwargs["seqLenUnroll"]
        dInnerUnroll = self.kwargs["dInnerUnroll"]
        dConv = self.kwargs["dConv"]
        convUnroll = self.kwargs["convUnroll"]

        assert convUnroll * NBIT == BANK_BYTES * 8, "switchCore output width must match 1 bank width"

        streamers = {
            "R0": (  # osCore in
                [
                    dModel,  # K
                    seqLen // seqLenUnroll,  # M
                    dInner // dInnerUnroll,  # N
                ],
                [
                    seqLenUnroll * NBIT // 8,
                    dModel * seqLenUnroll * NBIT // 8,
                    0,
                ],
            ),
            "R1": (  # oscore weight
                [
                    dModel,  # K
                    seqLen // seqLenUnroll,  # M
                    dInner // dInnerUnroll,  # N
                ],
                [
                    dInnerUnroll * NBIT // 8,
                    0,
                    dModel * dInnerUnroll * NBIT // 8,
                ],
            ),
            "R3": (  #  conv (switchCore) weight: layout is row-major [dInner, dConv]
                [dConv * dInner // convUnroll],
                [BANK_BYTES],
            ),
            "R4": (  #  conv (switchCore) bias: layout is row-major [dInner]
                [dInner // convUnroll],
                [BANK_BYTES],
            ),
            "R12": (  # iscore weight
                [
                    dModel,  # N
                    seqLen // seqLenUnroll,  # M
                    dInner // dInnerUnroll,  # K
                ],
                [
                    dInnerUnroll * NBIT // 8,
                    0,
                    dModel * dInnerUnroll * NBIT // 8,
                ],
            ),
            "R13": (  # isCore psum
                # First inject zeros, then (K-1) times the full output matrix
                # The initial values (C) can be at the same addresses as the output matrix
                [
                    (seqLen // seqLenUnroll) * dModel,  # one output matrix
                    dInner // dInnerUnroll,  # complete reduction dimension (K)
                ],
                [
                    seqLenUnroll * NBIT // 8,
                    0,  # Go to same addresses again
                ],
            ),
            "W1": (  # conv output
                [seqLen * dInner // convUnroll],
                [convUnroll * NBIT // 8],
            ),
            "W3": (  # isCore output: EXACTLY the same as psum reader R13
                [
                    (seqLen // seqLenUnroll) * dModel,
                    dInner // dInnerUnroll,
                ],
                [
                    seqLenUnroll * NBIT // 8,
                    0,
                ],
            ),
        }

        specs = [
            ("oscore_in", seqLen * dModel * NBIT // 8),
            ("oscore_weight", dModel * dInner * NBIT // 8),
            ("conv_weight", dInner * dConv * NBIT // 8),
            ("conv_bias", dInner * NBIT // 8),
            ("conv_out", seqLen * dInner * NBIT // 8),
            ("iscore_weight", dModel * dInner * NBIT // 8),
            ("iscore_out", seqLen * dModel * NBIT // 8),
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)
        scalars = {**lengths, **deltas}

        # Sampled outputs plus full tensor payloads.
        tests = {"conv_out": seqLen * dInner, "iscore_out": seqLen * dModel}

        test_data = {
            name: "uint16_t"
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

        self.build_mode(mode, streamers, scalars=scalars, test_data=test_data, tests=tests)

    def build_Phase2_data(self):
        mode = self.kwargs["M1_PHASE2"]
        seqLen = self.kwargs["seqLen"]
        dModel = self.kwargs["dModel"]
        dInner = self.kwargs["dInner"]
        dtRank = self.kwargs["dtRank"]
        dConv = self.kwargs["dConv"]
        seqLenUnroll = self.kwargs["seqLenUnroll"]
        dInnerUnroll = self.kwargs["dInnerUnroll"]
        dtRankUnroll = self.kwargs["dtRankUnroll"]
        convUnroll = self.kwargs["convUnroll"]
        dState = self.kwargs["dState"]
        delaySU = self.kwargs["delaySU"]
        switchcore_serial_width = self.kwargs["switchcore_serial_width"]
        oscore_serial_width = self.kwargs["oscore_serial_width"]
        iscore_serial_width = self.kwargs["iscore_serial_width"]
        suc_serial_width_A = self.kwargs["suc_serial_width_A"]
        suc_serial_width_BC = self.kwargs["suc_serial_width_BC"]  # Streamer width is 2x this value!

        assert convUnroll * NBIT == BANK_BYTES * 8, "switchCore output width must match 1 bank width"

        suc_parallel_widthA = dState * NBIT
        suc_parallel_widthBC = dState * NBIT
        oscore_array_width_d = seqLenUnroll * dInnerUnroll * NBIT
        iscore_array_width_d = seqLenUnroll * dInnerUnroll * NBIT

        # Reads out a layout that is stored in convFormat, in SUC format ordering
        bounds_conv_to_suc = [
            (convUnroll * seqLenUnroll) // (BANKWIDTH // NBIT),  # subTileSize / (elem per transfer)
            seqLen // seqLenUnroll,  # tiles per window
            dInnerUnroll // convUnroll,  # subtiles per tile
            dInner // dInnerUnroll,  # windows per tensor
        ]
        strides_conv_to_suc = [
            BANK_BYTES,
            seqLenUnroll * dInnerUnroll * NBIT // 8,  # tile size
            convUnroll * seqLenUnroll * NBIT // 8,  # subtile size
            seqLen * dInner * NBIT // 8,  # window size
        ]

        streamers = {
            "R0": (  # osCore in
                [
                    dModel,  # K
                    seqLen // seqLenUnroll,  # M
                    dInner // dInnerUnroll,  # N
                ],
                [
                    seqLenUnroll * NBIT // 8,
                    dModel * seqLenUnroll * NBIT // 8,
                    0,
                ],
            ),
            "R1": (  # oscore weight
                [
                    dModel,  # K
                    seqLen // seqLenUnroll,  # M
                    dInner // dInnerUnroll,  # N
                ],
                [
                    dInnerUnroll * NBIT // 8,
                    0,
                    dModel * dInnerUnroll * NBIT // 8,
                ],
            ),
            "R2": (  #  switchCore in (deltaMinor)
                [
                    dtRank // dtRankUnroll,  # K
                    seqLen,  # M
                    dInner // convUnroll,  # N
                ],
                [
                    BANK_BYTES,
                    (dtRank // dtRankUnroll) * BANK_BYTES,
                    0,
                ],
            ),
            "R3": (  #  switchCore weight (partition 1). Weights rotate internally so no reuse in seqLen here
                [(dtRank // dtRankUnroll) * (dInner // convUnroll)],
                [BANK_BYTES],
            ),
            "R4": (  #  switchCore bias
                [dInner // convUnroll],
                [BANK_BYTES],
            ),
            "R5": (  #  switchCore weight (partition 2)
                [(dtRank // dtRankUnroll) * (dInner // convUnroll)],
                [BANK_BYTES],
            ),
            "R6": (  # SUC A
                [dInner * (suc_parallel_widthA // suc_serial_width_A)],
                [suc_serial_width_A // 8],
            ),
            "R7": (  # SUC BC
                [
                    (suc_parallel_widthBC // suc_serial_width_BC),
                    seqLen,
                    dInner // delaySU,  # Irrelevant dimension
                ],
                [
                    (2 * suc_serial_width_BC) // 8,
                    (2 * suc_parallel_widthBC) // 8,
                    0,
                ],
            ),
            "R8": ([dInner], [BANK_BYTES]),  # SUC D
            "R9": (bounds_conv_to_suc, strides_conv_to_suc),  # SUC x.
            "R10": (bounds_conv_to_suc, strides_conv_to_suc),  # SUC z
            "R11": (  # iscore in. Stored in convFormat
                [(dInner // dInnerUnroll) * (seqLen // seqLenUnroll) * (iscore_array_width_d // iscore_serial_width)],
                [iscore_serial_width // 8],
            ),
            "R12": (  # iscore weight
                [
                    dModel,  # N
                    seqLen // seqLenUnroll,  # M
                    dInner // dInnerUnroll,  # K
                ],
                [
                    dInnerUnroll * NBIT // 8,
                    0,
                    dModel * dInnerUnroll * NBIT // 8,
                ],
            ),
            "R13": (  # isCore psum
                # First inject zeros, then (K-1) times the full output matrix
                # The initial values (C) can be at the same addresses as the output matrix
                [
                    (seqLen // seqLenUnroll) * dModel,  # one output matrix
                    dInner // dInnerUnroll,  # complete reduction dimension (K)
                ],
                [
                    seqLenUnroll * NBIT // 8,
                    0,  # Go to same addresses again
                ],
            ),
            "W0": (  # osCore out: writes in convFormat
                [(oscore_array_width_d // oscore_serial_width) * (seqLen // seqLenUnroll) * (dInner // dInnerUnroll)],
                [oscore_serial_width // 8],
            ),
            "W2": (  # SUC output y. Produced in SUC format, must be stored in convFormat
                bounds_conv_to_suc,
                strides_conv_to_suc,
            ),
            "W3": (  # isCore output: EXACTLY the same as psum reader R13
                [
                    (seqLen // seqLenUnroll) * dModel,
                    dInner // dInnerUnroll,
                ],
                [
                    seqLenUnroll * NBIT // 8,
                    0,
                ],
            ),
        }

        tensor_size = seqLen * dInner * NBIT // 8
        specs = [
            ("oscore_in", seqLen * dModel * NBIT // 8),
            ("oscore_weight", dModel * dInner * NBIT // 8),
            ("z", tensor_size),
            (
                "dt_in",
                self.pad_to_bankwidth(dInner * dtRank * NBIT, switchcore_serial_width) // 8,
            ),
            ("dt_weight_1", dInner * dConv * NBIT // 8),
            ("dt_weight_2", dInner * (dtRankUnroll - dConv) * NBIT // 8),
            ("dt_bias", dInner * NBIT // 8),
            ("x", tensor_size),
            ("A", dInner * dState * NBIT // 8),
            ("BC", 2 * seqLen * dState * NBIT // 8),
            ("D", dInner * NBIT // 8),
            ("y", tensor_size),
            ("iscore_weight", dModel * dInner * NBIT // 8),
            ("iscore_out", seqLen * dModel * NBIT // 8),
        ]

        lengths, deltas = self._collect_lengths_and_deltas(specs)
        scalars = {**lengths, **deltas}

        tests = {"z": seqLen * dInner, "y": seqLen * dInner, "iscore_out": seqLen * dModel}
        test_data = {
            name: "uint16_t"
            for name in (
                "oscore_in",
                "oscore_weight",
                "oscore_expected",  # aka matrix z
                "switchcore_in",
                "switchcore_weight_1",
                "switchcore_weight_2",
                # "suc_state",
                "suc_A",
                "suc_BC",
                "suc_D",
                "suc_x",
                "suc_expected",
                "iscore_weight",
                "iscore_bias",
                "iscore_expected",
            )
        }

        self.build_mode(mode, streamers, scalars=scalars, test_data=test_data, tests=tests)


def main():
    parser = argparse.ArgumentParser(description="Generate data for kernels")
    parser.add_argument(
        "--swcfg",
        type=pathlib.Path,
        required=True,
        help="Select param config file kernel",
    )
    parser.add_argument(
        "--hwcfg",
        type=pathlib.Path,
        required=True,
        help="Select hardware config file kernel",
    )
    args = parser.parse_args()

    # Load param config file
    with args.swcfg.open() as f:
        param = hjson.loads(f.read())

    # Load hardware config file # NOTE not used
    # with args.hwcfg.open() as f:
    #     hw = hjson.loads(f.read())

    # Merge dictionaries (hw overrides param in case of conflicts)
    merged_config = {**param}  # , **hw}

    # Emit header file
    generator = DataGenerator(**merged_config)
    print(generator.emit_header_file())


if __name__ == "__main__":
    main()
