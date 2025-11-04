#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>
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
        self.format_params()
        self.generate_OSGeMM_data()
        self.data.insert(0, "#include <stdint.h>\n\n")
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

    def _read_data_int(self, filename: str):
        """Read a vec from a file."""
        with open(os.path.join(DATA_OUT_DIR, filename), "r") as f:
            lines = f.readlines()
        data_lines = [line.strip() for line in lines if not line.startswith("#")]
        return [int(x) for x in data_lines]

    def format_test_sample_indices(self, num_outputs: int):
        """Format variables used to test only a subset of the output."""
        test_sample_count = min(num_outputs, TEST_SAMPLE_COUNT)
        self.format_int(test_sample_count)

        self.format_vector(
            "int32_t",
            "test_sample_indices",
            [random.randint(0, num_outputs - 1) for _ in range(test_sample_count)],
        )

    def format_channel_enable(self, streamer_name: str, total_channel_width: int):
        """If a streamer has more than 1 channel (memory port), it can disable some channels"""
        enabled_channel_CSR_num = int(math.ceil(total_channel_width / BANKWIDTH / 32))
        assert enabled_channel_CSR_num == 1, "The C code currently does not support having more than 32 channels"
        # If the channel is wide, it must be divisible by 8. If narrow, it must be divisible by 1.
        nb_bits = max(8, int(total_channel_width / BANKWIDTH + 7) // 8 * 8)
        channel_en = (1 << nb_bits) - 1
        self.format("int32_t", f"channel_en_{streamer_name}", channel_en)

    def generate_OSGeMM_data(self):
        # -------------------
        # Parameters
        # -------------------
        seqLen = self.kwargs["seqLen"]
        dModel = self.kwargs["dModel"]
        dInner = self.kwargs["dInner"]
        Mu = self.kwargs["Mu"]
        Nu = self.kwargs["Nu"]
        serial_width_d = self.kwargs["serial_width_d"]

        nbit_a = 16  # BF16. Hardcoded for now
        nbit_b = 16
        nbit_c = 16
        nbit_d = 16

        # In VersaCore naming convention
        M = seqLen // Mu
        K = dModel
        N = dInner // Nu
        self.format("uint32_t", "M", M)
        self.format("uint32_t", "K", K)
        self.format("uint32_t", "N", N)

        # Unrolled widths: parallel size of the serial-to-parallel converter
        a_array_width = Mu * nbit_a
        b_array_width = Nu * nbit_b
        c_array_width = Mu * Nu * nbit_c
        d_array_width = Mu * Nu * nbit_d
        assert c_array_width == d_array_width, "C and D array width must be the same"
        assert d_array_width % serial_width_d == 0, "d_array_width must be divisible by serial_width_d"

        data_length_a = M * K * Mu * nbit_a / 8
        data_length_b = K * N * Nu * nbit_b / 8
        data_length_c = M * N * Mu * Nu * nbit_c / 8
        data_length_d = M * N * Mu * Nu * nbit_d / 8
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
        self.format_channel_enable("A", a_array_width)

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
        self.format_channel_enable("B", b_array_width)
        # -------------------
        # Writer 0: output D
        # -------------------

        bounds_D = [d_array_width / serial_width_d * M * N]
        strides_D = [serial_width_d / 8]
        self.format_temporal_bounds_strides("D", bounds_D, strides_D)
        self.format_spatial_stride("D", BANKWIDTH // 8)
        self.format_channel_enable("D", serial_width_d)

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

        # Parse test data from external file.
        try:
            A_int = self._read_data_int("A.bin")
            B_int = self._read_data_int("B.bin")
            C_int = self._read_data_int("C.bin")
            D_int = self._read_data_int("D.bin")
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Error loading test data: {e}. Did you run the scala data generator and is the data directory correct?"
            )

        self.format_test_sample_indices(len(D_int))
        self.format_vector("uint16_t", "A", A_int)
        self.format_vector("uint16_t", "B", B_int)
        self.format_vector("uint16_t", "C", C_int)
        self.format_vector("uint16_t", "D", D_int)

    def generate_Phase1_data(self):
        # -------------------
        # Parameters
        # -------------------
        seqLen = self.kwargs["seqLen"]
        dModel = self.kwargs["dModel"]
        dInner = self.kwargs["dInner"]
        Mu = self.kwargs["Mu"]
        Nu = self.kwargs["Nu"]
        dConv = self.kwargs["dConv"]
        convUnroll = self.kwargs["convUnroll"]
        serial_width_d = self.kwargs["serial_width_d"]

        nbit_a = 16  # BF16. Hardcoded for now
        nbit_b = 16
        nbit_c = 16
        nbit_d = 16
        nbit = 16

        # In VersaCore naming convention
        M = seqLen // Mu
        K = dModel
        N = dInner // Nu
        self.format("uint32_t", "M", M)
        self.format("uint32_t", "K", K)
        self.format("uint32_t", "N", N)

        # Unrolled widths: parallel size of the serial-to-parallel converter
        b_array_width = Nu * nbit_b
        c_array_width = Mu * Nu * nbit_c
        d_array_width = Mu * Nu * nbit_d
        assert c_array_width == d_array_width, "C and D array width must be the same"
        assert d_array_width % serial_width_d == 0, "d_array_width must be divisible by serial_width_d"

        data_length_a = M * K * Mu * nbit_a / 8
        data_length_b = K * N * Nu * nbit_b / 8
        data_length_c = M * N * Mu * Nu * nbit_c / 8
        data_length_d = M * N * Mu * Nu * nbit_d / 8
        self.format_int(data_length_a)
        self.format_int(data_length_b)
        self.format_int(data_length_c)
        self.format_int(data_length_d)

        # ------------
        # Reader 0: in
        # ------------
        os_core_width_a = Mu * nbit_a
        bounds_R0 = [K, M, N]
        strides_R0 = [
            os_core_width_a / 8,
            K * os_core_width_a / 8,
            0,
        ]
        self.format_temporal_bounds_strides("R0", bounds_R0, strides_R0)
        self.format_spatial_stride("R0", BANKWIDTH // 8)
        self.format_channel_enable("R0", os_core_width_a)

        # ---------------------------------
        # Reader1: in proj weight (osCore)
        # ---------------------------------
        strides_R1 = [
            b_array_width / 8,  # K
            0,  # M - irrelevant dimension
            K * b_array_width / 8,  # N
        ]
        self.format_spatial_stride("R1", BANKWIDTH // 8)
        self.format_temporal_bounds_strides("R1", bounds_R0, strides_R1)

        # ---------------------------------
        # Reader 3: conv weight (switchCore)
        # ---------------------------------
        # layout is row-major [dInner, dConv]. 1 elem per cycle (currently 1 elem per bank)
        bounds_R3 = [dConv, dInner]
        strides_R3 = [BANKWIDTH // 8, dConv * BANKWIDTH // 8]
        self.format_spatial_stride("R3", BANKWIDTH // 8)
        self.format_temporal_bounds_strides("R3", bounds_R3, strides_R3)

        # ---------------------------------
        # Reader 4: conv bias (switchCore)
        # ---------------------------------
        # layout is [dInner]. 1 elem per cycle (currently 1 elem per bank)
        bounds_R4 = [dInner]
        strides_R4 = [BANKWIDTH // 8]
        self.format_spatial_stride("R4", BANKWIDTH // 8)
        self.format_temporal_bounds_strides("R4", bounds_R4, strides_R4)

        # ---------------------------------
        # Reader 12: out proj weight (isCore)
        # ---------------------------------

        # TODO
        bounds_R12 = [dInner]
        strides_R12 = [BANKWIDTH // 8]
        self.format_spatial_stride("R4", BANKWIDTH // 8)
        self.format_temporal_bounds_strides("R4", bounds_R12, strides_R12)

        # ---------------------------------
        # Reader 13: out proj psum (isCore)
        # ---------------------------------

        # TODO
        bounds_R13 = [dInner]
        strides_R13 = [BANKWIDTH // 8]
        self.format_spatial_stride("R4", BANKWIDTH // 8)
        self.format_temporal_bounds_strides("R4", bounds_R13, strides_R13)

        # -------------------
        # Writer 1: output conv
        # -------------------
        switch_core_width = convUnroll * nbit
        bounds_W1 = [seqLen * dInner / convUnroll]
        strides_W1 = [switch_core_width / 8]
        self.format_temporal_bounds_strides("W1", bounds_W1, strides_W1)
        self.format_spatial_stride("W1", BANKWIDTH // 8)

        # -------------------
        # Writer 3: out proj D (isCore)
        # -------------------

        # TODO
        bounds_W3 = [dInner, M, N]
        strides_W3 = [BANKWIDTH // 8, dInner * BANKWIDTH // 8, M * BANKWIDTH // 8]
        self.format_temporal_bounds_strides("W3", bounds_W3, strides_W3)
        self.format_spatial_stride("W1", BANKWIDTH // 8)

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

        # Parse test data from external file.
        try:
            A_int = self._read_data_int("A.bin")
            B_int = self._read_data_int("B.bin")
            C_int = self._read_data_int("C.bin")
            D_int = self._read_data_int("D.bin")
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Error loading test data: {e}. Did you run the scala data generator and is the data directory correct?"
            )

        self.format_test_sample_indices(len(D_int))
        self.format_vector("uint16_t", "A", A_int)
        self.format_vector("uint16_t", "B", B_int)
        self.format_vector("uint16_t", "C", C_int)
        self.format_vector("uint16_t", "D", D_int)


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
