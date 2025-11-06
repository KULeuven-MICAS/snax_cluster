#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Robin Geens <robin.geens@esat.kuleuven.be>


import argparse
import pathlib
import hjson
import sys
import os
import math
import inspect
import random

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402
from snax_utils import align_wide_addr  # noqa E402

DATA_OUT_DIR = os.path.join(os.path.dirname(__file__), "generated")
BANKWIDTH = 64
TEST_SAMPLE_COUNT = 25
NUM_LOOPS = 4  # NOTE this must match the hjson config


class DataGenerator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.data: list[str] = []

    def emit_header_file(self):
        self.data.append("#include <stdint.h>\n")
        self.format_params()
        self.format("uint32_t", "test_sample_count", TEST_SAMPLE_COUNT)
        self.format("uint32_t", "channel_en", (1 << 32) - 1)  # Use global channel_en for all streamers
        self.generate_Phase1_data()
        self.generate_OSGeMM_data()
        return "\n".join(self.data)

    def format(self, type: str, var_name: str, value: int):
        self.data.append(format_scalar_definition(type, var_name, value))

    def format_int(self, value: int):
        """Format `value` as integer. The name will be the variable name in the caller's scope."""
        callers_local_vars = (
            inspect.currentframe().f_back.f_locals.items()  # pyright: ignore[reportOptionalMemberAccess]
        )
        variable_name = next((name for name, val in callers_local_vars if val is value), None)
        assert variable_name is not None, "Variable name not found"
        self.format("uint32_t", variable_name, value)

    def format_params(self):
        """Takes all parameters from the kwargs and formats them as integers."""
        for key, value in self.kwargs.items():
            self.format("uint32_t", key, value)

    def format_vector(self, type: str, var_name: str, value: list[int]):
        self.data.append(format_vector_definition(type, var_name, value))

    def read_and_format_vector(self, type: str, tensor_name: str):
        try:
            tensor_data = self._read_data_int(tensor_name)
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Error loading test data for tensor {tensor_name}: {e}. Did you run the scala data generator and is the data directory correct?"
            )
        self.format_vector(type, tensor_name, tensor_data)

    def format_temporal_bounds_strides(self, streamer_name: str, bounds: list[int], strides: list[int]):
        """Format temporal bounds and strides for a streamer by automatically naming the variables and adding defaults.
        bounds are from inner to outer loop.
        """
        # Extend with defaults
        bounds = bounds + [1] * (NUM_LOOPS - len(bounds))
        strides = strides + [0] * (NUM_LOOPS - len(strides))
        # Save each bound/stride as a separate variable
        for i in range(NUM_LOOPS):
            self.format("uint32_t", f"{streamer_name}tlbound{i}", bounds[i])
            self.format("uint32_t", f"{streamer_name}tlstride{i}", strides[i])
        # Group values into array
        self.data += [f"int32_t {streamer_name}tlbound[] = {{{', '.join(map(str, bounds))}}};"]
        self.data += [f"int32_t {streamer_name}tlstride[] = {{{', '.join(map(str, strides))}}};"]

    def format_spatial_stride(self, streamer_name: str, stride: int):
        self.format("uint32_t", f"{streamer_name}slstride0", stride)
        self.data += [f"int32_t {streamer_name}slstride[] = {{{stride}}};"]

    def _read_data_int(self, tensor_name: str):
        """Read a vec from a file."""
        with open(os.path.join(DATA_OUT_DIR, tensor_name + ".bin"), "r") as f:
            lines = f.readlines()
        data_lines = [line.strip() for line in lines if not line.startswith("#")]
        return [int(x) for x in data_lines]

    def format_test_sample_indices(self, tensor_name: str, tensor_size: int):
        """Format variables used to test only a subset of the output."""
        self.format_vector(
            "int32_t",
            f"test_sample_indices_{tensor_name}",
            [random.randint(0, tensor_size - 1) for _ in range(TEST_SAMPLE_COUNT)],
        )

    def format_channel_enable(self, streamer_name: str):
        """If a streamer has more than 1 channel (memory port), it can disable some channels.
        Here, we ignore this and set all channels to 1."""
        channel_en = (1 << 32) - 1
        self.format("uint32_t", f"channel_en_{streamer_name}", channel_en)

    def generate_OSGeMM_data(self):
        # -------------------
        # Parameters
        # -------------------
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
        self.format_temporal_bounds_strides("A", bounds_A, strides_A)
        # INFO [RG] Spatial stride is 1 full bank: i.e. streamer accesses multiple, sequential banks.
        self.format_spatial_stride("A", BANKWIDTH // 8)
        self.format_channel_enable("A")

        # ------------------
        # Reader 1: weight B
        # ------------------

        strides_B = [
            b_array_width / 8,  # K
            0,  # M - irrelevant dimension
            K * b_array_width / 8,  # N
        ]
        self.format_spatial_stride("B", BANKWIDTH // 8)
        self.format_temporal_bounds_strides("B", bounds_A, strides_B)
        self.format_channel_enable("B")
        # -------------------
        # Writer 0: output D
        # -------------------

        bounds_D = [d_array_width / serial_width_d * M * N]
        strides_D = [serial_width_d / 8]
        self.format_temporal_bounds_strides("D", bounds_D, strides_D)
        self.format_spatial_stride("D", BANKWIDTH // 8)
        self.format_channel_enable("D")

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
        self.format_test_sample_indices("D", tensor_size=seqLen * dInner)
        self.read_and_format_vector("uint16_t", "A")
        self.read_and_format_vector("uint16_t", "B")
        self.read_and_format_vector("uint16_t", "C")
        self.read_and_format_vector("uint16_t", "D")

    def generate_Phase1_data(self):
        # -------------------
        # Parameters
        # -------------------
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
        self.format_temporal_bounds_strides("R0", bounds_R0, strides_R0)
        self.format_spatial_stride("R0", BANKWIDTH // 8)
        self.format_channel_enable("R0")

        # ----------------------
        # Reader1: osCore weight
        # ----------------------
        strides_R1 = [
            (dInnerUnroll * nbit) / 8,  # K
            0,  # M - irrelevant dimension
            dModel * (dInnerUnroll * nbit) / 8,  # N
        ]
        self.format_spatial_stride("R1", BANKWIDTH // 8)
        self.format_temporal_bounds_strides("R1", bounds_R0, strides_R1)
        self.format_channel_enable("R1")

        # ---------------------------------
        # Reader 3: conv (switchCore) weight
        # ---------------------------------
        # layout is row-major [dInner, dConv]
        bounds_R3 = [dConv * dInner // convUnroll]
        strides_R3 = [BANKWIDTH // 8]
        self.format_spatial_stride("R3", BANKWIDTH // 8)
        self.format_temporal_bounds_strides("R3", bounds_R3, strides_R3)
        self.format_channel_enable("R3")

        # ---------------------------------
        # Reader 4: conv (switchCore) bias
        # ---------------------------------
        bounds_R4 = [dInner // convUnroll]
        strides_R4 = [BANKWIDTH // 8]
        self.format_spatial_stride("R4", BANKWIDTH // 8)
        self.format_temporal_bounds_strides("R4", bounds_R4, strides_R4)
        self.format_channel_enable("R4")

        # ------------------------
        # Reader 12: isCore weight
        # ------------------------
        bounds_R12 = [
            dModel,  # N
            seqLen // seqLenUnroll,  # M
            dInner // dInnerUnroll,  # K
        ]
        strides_R12 = [
            (dInnerUnroll * nbit) // 8,  # TODO versacore heeft de strides gewisseld?
            0,
            dModel * (dInnerUnroll * nbit) // 8,
        ]
        self.format_spatial_stride("R12", BANKWIDTH // 8)
        self.format_temporal_bounds_strides("R12", bounds_R12, strides_R12)

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
        self.format_spatial_stride("R13", BANKWIDTH // 8)
        self.format_temporal_bounds_strides("R13", bounds_R13, strides_R13)
        self.format_channel_enable("R13")

        # -------------------
        # Writer 1: output conv
        # -------------------
        assert (convUnroll * nbit) == BANKWIDTH, "switchCore output width must be equal to BANKWIDTH"
        bounds_W1 = [seqLen * dInner // convUnroll]
        strides_W1 = [(convUnroll * nbit) // 8]
        self.format_temporal_bounds_strides("W1", bounds_W1, strides_W1)
        # TODO What if we don't set spatial stride in 1-port streamer?
        self.format_spatial_stride("W1", BANKWIDTH // 8)
        self.format_channel_enable("W1")

        # -----------------------
        # Writer 3: isCore output
        # -----------------------
        # Exactly the same as the psum reader
        self.format_temporal_bounds_strides("W3", bounds_R13, strides_R13)
        self.format_spatial_stride("W3", BANKWIDTH // 8)

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
        self.format_int(delta_oscore_in)
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
        self.format_test_sample_indices("conv_out", tensor_size=seqLen * dInner)
        self.format_test_sample_indices("iscore_out", tensor_size=seqLen * dModel)
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
