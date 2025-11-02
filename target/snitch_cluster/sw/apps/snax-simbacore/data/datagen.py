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

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402
from snax_utils import align_wide_addr  # noqa E402

DATA_OUT_DIR = os.path.join(os.path.dirname(__file__), "generated")
BANKWIDTH = 64


class DataGenerator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.data: list[str] = []

    def emit_header_file(self):
        self.format_params()
        self.generate_OSGeMM_data()
        self.data.insert(0, "#include <stdint.h>\n\n")
        return "\n".join(self.data)

    def _gen_channel_enable_CSR(self, channel_en_CSR, channel_en_bits):
        for i in range(channel_en_bits):
            element_index = i // 32  # Determine which element to modify
            bit_position = i % 32  # Position within the element
            if element_index < len(channel_en_CSR):
                channel_en_CSR[element_index] |= 1 << (bit_position)

        channel_en_CSR = [int(x) for x in channel_en_CSR][::-1]
        return channel_en_CSR

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

    def format_temporal_bounds_strides(self, streamer_name: str, bounds: list[int], strides: list[int], num_loops: int):
        """Format temporal bounds and strides for a streamer by automatically naming the variables and adding defaults.
        From inner to outer loop.
        """
        # Extend with defaults
        bounds = bounds + [1] * (num_loops - len(bounds))
        strides = strides + [0] * (num_loops - len(strides))
        for i in range(num_loops):
            self.format("uint32_t", f"{streamer_name}tlbound{i}", bounds[i])
            self.format("uint32_t", f"{streamer_name}tlstride{i}", strides[i])

    def format_spatial_stride(self, streamer_name: str, stride: int):
        self.format("uint32_t", f"{streamer_name}slstride0", stride)

    def _read_data_int(self, filename: str):
        """Read a vec from a file."""
        with open(os.path.join(DATA_OUT_DIR, filename), "r") as f:
            lines = f.readlines()
        data_lines = [line.strip() for line in lines if not line.startswith("#")]
        return [int(x) for x in data_lines]

    def generate_OSGeMM_data(self):
        # -------------------
        # Parameters
        # -------------------
        num_loops = 4  # NOTE this must match the hjson config
        M = self.kwargs["M"]
        K = self.kwargs["K"]
        N = self.kwargs["N"]
        Mu = self.kwargs["Mu"]
        Ku = self.kwargs["Ku"]
        Nu = self.kwargs["Nu"]
        serial_width_d = self.kwargs["serial_width_d"]
        nbit_a = 16  # BF16. Hardcoded for now
        nbit_b = 16
        nbit_c = 16
        nbit_d = 16

        # Unrolled widths: parallel size of the serial-to-parallel converter
        a_array_width = Mu * Ku * nbit_a
        b_array_width = Ku * Nu * nbit_b
        c_array_width = Mu * Nu * nbit_c
        d_array_width = Mu * Nu * nbit_d
        assert c_array_width == d_array_width, "C and D array width must be the same"
        assert d_array_width % serial_width_d == 0, "d_array_width must be divisible by serial_width_d"

        data_length_a = M * K * Mu * Ku * nbit_a / 8
        data_length_b = K * N * Ku * Nu * nbit_b / 8
        data_length_c = M * N * Mu * Nu * nbit_c / 8
        data_length_d = M * N * Mu * Nu * nbit_d / 8

        self.format_int(data_length_a)
        self.format_int(data_length_b)
        self.format_int(data_length_c)
        self.format_int(data_length_d)

        # -------------------
        # A streamer setting
        # -------------------

        # INFO [RG] Number of lines here corresponds to "temporal_dim" in the hjson file.
        # for n in N (irrelevant dimension)
        #   for m in M
        #     for k in K
        #         parfor s in (tileSize / bankWidth)
        #             addr = k * tile_size + m * K * tile_size + s * bankWidth
        bounds_A = [
            K,
            M,
            N,
        ]
        strides_A = [
            a_array_width / 8,
            K * a_array_width / 8,
            0,
        ]
        self.format_temporal_bounds_strides("A", bounds_A, strides_A, num_loops)
        # INFO [RG] Spatial stride is 1 full bank: i.e. streamer accesses multiple, sequential banks.
        self.format_spatial_stride("A", BANKWIDTH // 8)

        # TODO [RG] what does this mean exactly?
        A_enabled_channel_CSR_num = int(math.ceil(a_array_width / BANKWIDTH / 32))
        channel_en_A = [0] * A_enabled_channel_CSR_num
        # Related to if this is a wide channel or not. If wide, must be divisible by 8, if narrow, must be divisible by 1
        channel_en_A_bits = max(8, int((Mu * Ku * nbit_a / BANKWIDTH + 7) // 8 * 8))
        channel_en_A = self._gen_channel_enable_CSR(channel_en_A, channel_en_A_bits)
        self.data += ["int32_t channel_en_A[] = { " + ", ".join(map(str, channel_en_A)) + " };"]

        # -------------------
        # B streamer setting
        # -------------------

        strides_B = [
            b_array_width / 8,  # K
            0,  # M - irrelevant dimension
            K * b_array_width / 8,  # N
        ]
        self.format_spatial_stride("B", BANKWIDTH // 8)
        self.format_temporal_bounds_strides("B", bounds_A, strides_B, num_loops)

        B_enabled_channel_CSR_num = int(math.ceil(b_array_width / BANKWIDTH / 32))
        channel_en_B = [0] * B_enabled_channel_CSR_num
        channel_en_B_bits = max(8, int((Nu * Ku * nbit_b / BANKWIDTH + 7) // 8 * 8))
        channel_en_B = self._gen_channel_enable_CSR(channel_en_B, channel_en_B_bits)
        self.data += ["int32_t channel_en_B[] = { " + ", ".join(map(str, channel_en_B)) + " };"]

        # -------------------
        # D streamer setting
        # -------------------
        # spatial settings

        # Temporal settings. Order: N, M, K

        bounds_D = [
            d_array_width / serial_width_d,  # serDesFactor
            M,
            N,  # No K here because output stationary
        ]
        strides_D = [
            serial_width_d / 8,
            d_array_width / 8,
            M * d_array_width / 8,
        ]
        self.format_temporal_bounds_strides("D", bounds_D, strides_D, num_loops)
        self.format_spatial_stride("D", BANKWIDTH // 8)

        D_enabled_channel_CSR_num = int(math.ceil(serial_width_d / BANKWIDTH / 32))
        channel_en_D = [0] * D_enabled_channel_CSR_num
        channel_en_D_bits = int((Mu * Nu * nbit_c / BANKWIDTH + 7) // 8 * 8)
        channel_en_D = self._gen_channel_enable_CSR(channel_en_D, channel_en_D_bits)
        self.data += ["int32_t channel_en_D[] = { " + ", ".join(map(str, channel_en_D)) + " };"]

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
        # TODO these file names are hardcoded too much
        try:
            A_int = self._read_data_int("A.bin")
            B_int = self._read_data_int("B.bin")
            C_int = self._read_data_int("C.bin")
            D_int = self._read_data_int("D.bin")
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Error loading test data: {e}. Did you run the scala data generator and is the data directory correct?"
            )

        self.format_vector("uint16_t", "A", A_int)
        self.format_vector("uint16_t", "B", B_int)
        self.format_vector("uint16_t", "C", C_int)
        self.format_vector("uint16_t", "D", D_int)

        # NOTE [RG] why do whe need this
        self.format("int32_t", "set_addr_remap_index_A", 0)
        self.format("int32_t", "set_addr_remap_index_B", 0)
        self.format("int32_t", "set_addr_remap_index_C", 0)
        self.format("int32_t", "set_addr_remap_index_D", 0)


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
