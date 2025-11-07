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
from snax_utils import align_wide_addr  # noqa: E402

# try:
#     from .datagen_util import DataGeneratorBase, BANKWIDTH  # type: ignore[import]
# except ImportError:  # pragma: no cover - executed when run as a script
from datagen_util import DataGeneratorBase  # type: ignore[import]


NB_TEST_SAMPLES = 25
BANKWIDTH = 64


class DataGenerator(DataGeneratorBase):
    def run(self):
        self.gen_common_data()
        self.generate_OSGeMM_data()
        self.generate_Phase1_data()

    def gen_common_data(self):
        self.format("uint32_t", "nb_test_samples", NB_TEST_SAMPLES)
        self.format("uint32_t", "channel_en", (1 << 32) - 1)  # Use global channel_en for all streamers
        self.format_params()

    def generate_OSGeMM_data(self):
        # -------------------
        # Parameters
        # -------------------
        mode = self.kwargs["M2_OSGEMM"]  # mode bit = 2
        seqLen = self.kwargs["seqLen"]
        dModel = self.kwargs["dModel"]
        dInner = self.kwargs["dInner"]
        Mu = self.kwargs["seqLenUnroll"]
        Nu = self.kwargs["dInnerUnroll"]
        serial_width_d = self.kwargs["serial_width_d"]

        nbit = 16  # BF16. Hardcoded for now

        # In VersaCore naming convention
        M = seqLen // Mu
        K = dModel
        N = dInner // Nu

        # Unrolled widths: parallel size of the serial-to-parallel converter
        a_array_width = Mu * nbit
        b_array_width = Nu * nbit
        c_array_width = Mu * Nu * nbit
        d_array_width = Mu * Nu * nbit
        assert c_array_width == d_array_width, "C and D array width must be the same"
        assert d_array_width % serial_width_d == 0, "d_array_width must be divisible by serial_width_d"

        data_length_a = M * K * Mu * nbit / 8
        data_length_b = K * N * Nu * nbit / 8
        data_length_c = M * N * Mu * Nu * nbit / 8
        data_length_d = M * N * Mu * Nu * nbit / 8
        self.format_int(data_length_a)
        self.format_int(data_length_b)
        self.format_int(data_length_c)
        self.format_int(data_length_d)

        # -----------------
        # Reader 0: input A
        # -----------------

        # INFO [RG] Number of lines here corresponds to "temporal_dim" in the hjson file.
        # for n in N (irrelevant dimension)
        #   for m in M
        #     for k in K
        #         parfor s in (tileSize / bankWidth)
        #             addr = k * tile_size + m * K * tile_size + s * bankWidth
        bounds_A = [K, M, N]
        strides_A = [
            a_array_width / 8,
            K * a_array_width / 8,
            0,
        ]
        self.format_temporal_bounds_strides("A", mode, bounds_A, strides_A)
        # INFO [RG] Spatial stride is 1 full bank: i.e. streamer accesses multiple, sequential banks.
        self.format_spatial_stride("A", mode, BANKWIDTH // 8)
        self.enable_channel("A", mode)

        # ------------------
        # Reader 1: weight B
        # ------------------
        strides_B = [
            b_array_width / 8,  # K
            0,  # M - irrelevant dimension
            K * b_array_width / 8,  # N
        ]
        self.format_spatial_stride("B", mode, BANKWIDTH // 8)
        self.format_temporal_bounds_strides("B", mode, bounds_A, strides_B)
        self.enable_channel("B", mode)

        # -------------------
        # Writer 0: output D
        # -------------------
        bounds_D = [d_array_width / serial_width_d * M * N]
        strides_D = [serial_width_d / 8]
        self.format_temporal_bounds_strides("D", mode, bounds_D, strides_D)
        self.format_spatial_stride("D", mode, BANKWIDTH // 8)
        self.enable_channel("D", mode)

        # ------------
        # Base address
        # ------------
        # Start address of the data (relative to some base address)
        delta_local_a = 0
        delta_local_b = align_wide_addr(delta_local_a + data_length_a)
        delta_local_c = align_wide_addr(delta_local_b + data_length_b)
        delta_local_d = align_wide_addr(delta_local_c + data_length_c)
        self.format_int(delta_local_a)
        self.format_int(delta_local_b)
        self.format_int(delta_local_c)
        self.format_int(delta_local_d)

        # -------------------
        # Test Data generation
        # -------------------

        # Parse test data from external file and format
        self.format_test_samples("D", tensor_size=seqLen * dInner, nb_test_samples=NB_TEST_SAMPLES)
        self.read_and_format_vector("uint16_t", "A")
        self.read_and_format_vector("uint16_t", "B")
        self.read_and_format_vector("uint16_t", "C")
        self.read_and_format_vector("uint16_t", "D")

    def generate_Phase1_data(self):
        # -------------------
        # Parameters
        # -------------------
        mode = self.kwargs["M0_PHASE1"]  # mode bit = 1
        seqLen = self.kwargs["seqLen"]
        dModel = self.kwargs["dModel"]
        dInner = self.kwargs["dInner"]
        seqLenUnroll = self.kwargs["seqLenUnroll"]
        dInnerUnroll = self.kwargs["dInnerUnroll"]
        dConv = self.kwargs["dConv"]
        convUnroll = self.kwargs["convUnroll"]

        nbit = 16  # BF16. Hardcoded for now

        # ------------
        # Reader 0: in
        # ------------
        bounds_R0 = [
            dModel,  # K
            seqLen // seqLenUnroll,  # M
            dInner // dInnerUnroll,  # N
        ]
        strides_R0 = [
            (seqLenUnroll * nbit) / 8,
            dModel * (seqLenUnroll * nbit) / 8,
            0,
        ]
        self.format_temporal_bounds_strides("R0", mode, bounds_R0, strides_R0)
        self.format_spatial_stride("R0", mode, BANKWIDTH // 8)
        self.enable_channel("R0", mode)

        # ----------------------
        # Reader1: osCore weight
        # ----------------------
        strides_R1 = [
            (dInnerUnroll * nbit) / 8,  # K
            0,  # M - irrelevant dimension
            dModel * (dInnerUnroll * nbit) / 8,  # N
        ]
        self.format_spatial_stride("R1", mode, BANKWIDTH // 8)
        self.format_temporal_bounds_strides("R1", mode, bounds_R0, strides_R1)
        self.enable_channel("R1", mode)

        # ---------------------------------
        # Reader 3: conv (switchCore) weight
        # ---------------------------------
        # layout is row-major [dInner, dConv]
        bounds_R3 = [dConv * dInner // convUnroll]
        strides_R3 = [BANKWIDTH // 8]
        self.format_spatial_stride("R3", mode, BANKWIDTH // 8)
        self.format_temporal_bounds_strides("R3", mode, bounds_R3, strides_R3)
        self.enable_channel("R3", mode)

        # ---------------------------------
        # Reader 4: conv (switchCore) bias
        # ---------------------------------
        bounds_R4 = [dInner // convUnroll]
        strides_R4 = [BANKWIDTH // 8]
        self.format_spatial_stride("R4", mode, BANKWIDTH // 8)
        self.format_temporal_bounds_strides("R4", mode, bounds_R4, strides_R4)
        self.enable_channel("R4", mode)

        # ------------------------
        # Reader 12: isCore weight
        # ------------------------
        bounds_R12 = [
            dModel,  # N
            seqLen // seqLenUnroll,  # M
            dInner // dInnerUnroll,  # K
        ]
        strides_R12 = [
            (dInnerUnroll * nbit) // 8,
            0,
            dModel * (dInnerUnroll * nbit) // 8,
        ]
        self.format_spatial_stride("R12", mode, BANKWIDTH // 8)
        self.format_temporal_bounds_strides("R12", mode, bounds_R12, strides_R12)
        self.enable_channel("R12", mode)

        # ----------------------
        # Reader 13: isCore psum
        # ----------------------

        # First inject zeros, then (K-1) times the full output matrix
        # I guess the initial values (C) can be at the same addresses as the output matrix
        bounds_R13 = [
            (seqLen // seqLenUnroll) * dModel,  # one output matrix
            (dInner // dInnerUnroll),  # complete reduction dimension
        ]
        strides_R13 = [
            (seqLenUnroll * nbit) // 8,
            0,  # Go to same addresses again
        ]
        self.format_spatial_stride("R13", mode, BANKWIDTH // 8)
        self.format_temporal_bounds_strides("R13", mode, bounds_R13, strides_R13)
        self.enable_channel("R13", mode)

        # -------------------
        # Writer 1: output conv
        # -------------------
        assert (convUnroll * nbit) == BANKWIDTH, "switchCore output width must be equal to BANKWIDTH"
        bounds_W1 = [seqLen * dInner // convUnroll]
        strides_W1 = [(convUnroll * nbit) // 8]
        self.format_temporal_bounds_strides("W1", mode, bounds_W1, strides_W1)
        # TODO What if we don't set spatial stride in 1-port streamer?
        self.format_spatial_stride("W1", mode, BANKWIDTH // 8)
        self.enable_channel("W1", mode)

        # -----------------------
        # Writer 3: isCore output
        # -----------------------
        # Exactly the same as the psum reader
        self.format_temporal_bounds_strides("W3", mode, bounds_R13, strides_R13)
        self.format_spatial_stride("W3", mode, BANKWIDTH // 8)
        self.enable_channel("W3", mode)

        # ------------
        # Base address
        # ------------
        # Matrix sizes
        length_oscore_in = seqLen * dModel * nbit / 8
        length_oscore_weight = dModel * dInner * nbit / 8
        length_conv_weight = dInner * dConv * nbit / 8
        length_conv_bias = dInner * nbit / 8
        length_conv_out = seqLen * dInner * nbit / 8
        length_iscore_weight = dModel * dInner * nbit / 8
        length_iscore_out = seqLen * dModel * nbit / 8
        length_iscore_bias = seqLen * dModel * nbit / 8
        self.format_int(length_oscore_in)
        self.format_int(length_oscore_weight)
        self.format_int(length_conv_weight)
        self.format_int(length_conv_bias)
        self.format_int(length_conv_out)
        self.format_int(length_iscore_weight)
        self.format_int(length_iscore_bias)
        self.format_int(length_iscore_out)

        # Address offsets
        delta_oscore_in = 0
        delta_oscore_weight = align_wide_addr(delta_oscore_in + length_oscore_in)
        delta_conv_weight = align_wide_addr(delta_oscore_weight + length_oscore_weight)
        delta_conv_bias = align_wide_addr(delta_conv_weight + length_conv_weight)
        delta_conv_out = align_wide_addr(delta_conv_bias + length_conv_bias)
        delta_iscore_weight = align_wide_addr(delta_conv_out + length_conv_out)
        delta_iscore_out = align_wide_addr(delta_iscore_weight + length_iscore_weight)
        self.format("uint32_t", "delta_oscore_in", delta_oscore_in)
        self.format_int(delta_oscore_weight)
        self.format_int(delta_conv_weight)
        self.format_int(delta_conv_bias)
        self.format_int(delta_conv_out)
        self.format_int(delta_iscore_weight)
        self.format_int(delta_iscore_out)

        # -------------------
        # Test Data generation
        # -------------------

        # Parse test data from external file. NOTE the tensor names must match those in Scala
        self.format_test_samples("conv_out", tensor_size=seqLen * dInner, nb_test_samples=NB_TEST_SAMPLES)
        self.format_test_samples("iscore_out", tensor_size=seqLen * dModel, nb_test_samples=NB_TEST_SAMPLES)
        self.read_and_format_vector("uint16_t", "oscore_in")
        self.read_and_format_vector("uint16_t", "oscore_weight")
        self.read_and_format_vector("uint16_t", "conv_weight")
        self.read_and_format_vector("uint16_t", "conv_bias")
        self.read_and_format_vector("uint16_t", "conv_out")  # golden vector
        self.read_and_format_vector("uint16_t", "iscore_weight")
        self.read_and_format_vector("uint16_t", "iscore_bias")  # psum initial value
        self.read_and_format_vector("uint16_t", "iscore_out")  # golden vector


def main():
    # Parsing cmd args
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
