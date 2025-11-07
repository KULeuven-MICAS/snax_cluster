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
import re

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from snax_utils import align_wide_addr  # noqa: E402

# try:
#     from .datagen_util import DataGeneratorBase, BANKWIDTH  # type: ignore[import]
# except ImportError:  # pragma: no cover - executed when run as a script
from datagen_util import DataGeneratorBase, BANK_BYTES  # type: ignore[import]


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

        data_length_a = M * K * a_array_width // 8
        data_length_b = K * N * b_array_width // 8
        data_length_c = M * N * d_array_width // 8
        data_length_d = data_length_c

        delta_a = 0
        delta_b = align_wide_addr(delta_a + data_length_a)
        delta_c = align_wide_addr(delta_b + data_length_b)
        delta_d = align_wide_addr(delta_c + data_length_c)

        scalars = {
            "data_length_a": data_length_a,
            "data_length_b": data_length_b,
            "data_length_c": data_length_c,
            "data_length_d": data_length_d,
            "delta_a": delta_a,
            "delta_b": delta_b,
            "delta_c": delta_c,
            "delta_d": delta_d,
        }

        test_data = {name: "uint16_t" for name in ("A", "B", "C", "D")}
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

        length_oscore_in = seqLen * dModel * NBIT // 8
        length_oscore_weight = dModel * dInner * NBIT // 8
        length_conv_weight = dInner * dConv * NBIT // 8
        length_conv_bias = dInner * NBIT // 8
        length_conv_out = seqLen * dInner * NBIT // 8
        length_iscore_weight = dModel * dInner * NBIT // 8
        length_iscore_bias = seqLen * dModel * NBIT // 8
        length_iscore_out = seqLen * dModel * NBIT // 8

        delta_oscore_in = 0
        delta_oscore_weight = align_wide_addr(delta_oscore_in + length_oscore_in)
        delta_conv_weight = align_wide_addr(delta_oscore_weight + length_oscore_weight)
        delta_conv_bias = align_wide_addr(delta_conv_weight + length_conv_weight)
        delta_conv_out = align_wide_addr(delta_conv_bias + length_conv_bias)
        delta_iscore_weight = align_wide_addr(delta_conv_out + length_conv_out)
        delta_iscore_out = align_wide_addr(delta_iscore_weight + length_iscore_weight)

        scalars = {
            "length_oscore_in": length_oscore_in,
            "length_oscore_weight": length_oscore_weight,
            "length_conv_weight": length_conv_weight,
            "length_conv_bias": length_conv_bias,
            "length_conv_out": length_conv_out,
            "length_iscore_weight": length_iscore_weight,
            "length_iscore_bias": length_iscore_bias,
            "length_iscore_out": length_iscore_out,
            "delta_oscore_in": delta_oscore_in,
            "delta_oscore_weight": delta_oscore_weight,
            "delta_conv_weight": delta_conv_weight,
            "delta_conv_bias": delta_conv_bias,
            "delta_conv_out": delta_conv_out,
            "delta_iscore_weight": delta_iscore_weight,
            "delta_iscore_out": delta_iscore_out,
        }

        # Sampled outputs plus full tensor payloads.
        tests = {"conv_out": seqLen * dInner, "iscore_out": seqLen * dModel}

        test_data = {
            "oscore_in": "uint16_t",
            "oscore_weight": "uint16_t",
            "conv_weight": "uint16_t",
            "conv_bias": "uint16_t",
            "conv_out": "uint16_t",
            "iscore_weight": "uint16_t",
            "iscore_bias": "uint16_t",
            "iscore_out": "uint16_t",
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
