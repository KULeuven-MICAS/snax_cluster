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

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402
from snax_utils import align_wide_addr  # noqa E402

DATA_OUT_DIR = os.path.join(os.path.dirname(__file__), "generated")
BANKWIDTH = 64


# Add stdint.h header
def emit_header_file(**kwargs):
    emit_str = "#include <stdint.h>\n\n"
    emit_str += emit_matmul_data(**kwargs)
    return emit_str


def gen_channel_enable_CSR(channel_en_CSR, channel_en_bits):
    for i in range(channel_en_bits):
        element_index = i // 32  # Determine which element to modify
        bit_position = i % 32  # Position within the element
        if element_index < len(channel_en_CSR):
            channel_en_CSR[element_index] |= 1 << (bit_position)

    channel_en_CSR = [int(x) for x in channel_en_CSR][::-1]
    return channel_en_CSR


# FP8 (E5M2) constants
FP8_EXP_BITS = 5
FP8_MAN_BITS = 2
FP8_EXP_BIAS = (2 ** (FP8_EXP_BITS - 1)) - 1  # bias = 15
FP8_MAX_EXP = (2**FP8_EXP_BITS) - 2  # 30 (reserve 31 for Inf/NaN)
FP8_MIN_EXP = 1  # exponent=0 is subnormal/zero


def emit_matmul_data(**kwargs):

    # -----------------
    # Workload settings
    # -----------------
    data_str = []

    M = kwargs["M"]
    K = kwargs["K"]
    N = kwargs["N"]
    mode = kwargs["mode"]

    data_str += [format_scalar_definition("uint32_t", "M", M)]
    data_str += [format_scalar_definition("uint32_t", "K", K)]
    data_str += [format_scalar_definition("uint32_t", "N", N)]
    data_str += [format_scalar_definition("uint32_t", "mode", mode)]

    # -------------------
    # Hardware parameters
    # -------------------
    Mu = kwargs["Mu"]
    Ku = kwargs["Ku"]
    Nu = kwargs["Nu"]
    data_str += [format_scalar_definition("uint32_t", "Mu", Mu)]
    data_str += [format_scalar_definition("uint32_t", "Ku", Ku)]
    data_str += [format_scalar_definition("uint32_t", "Nu", Nu)]

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
    serial_width_d = kwargs["serial_width_d"]

    data_length_a = M * K * Mu * Ku * nbit_a / 8
    data_length_b = K * N * Ku * Nu * nbit_b / 8
    data_length_c = M * N * Mu * Nu * nbit_c / 8
    data_length_d = M * N * Mu * Nu * nbit_d / 8
    data_str += [format_scalar_definition("int32_t", "data_length_a", data_length_a)]
    data_str += [format_scalar_definition("int32_t", "data_length_b", data_length_b)]
    data_str += [format_scalar_definition("int32_t", "data_length_c", data_length_c)]
    data_str += [format_scalar_definition("int32_t", "data_length_d", data_length_d)]

    # -------------------
    # A streamer setting
    # -------------------

    # INFO [RG] Spatial stride is 1 full bank: i.e. streamer accesses multiple, sequential banks.
    data_str += [format_scalar_definition("int32_t", "Aslstride0", BANKWIDTH / 8)]

    # INFO [RG] Number of lines here corresponds to "temporal_dim" in the hjson file.
    # for n in N (irrelevant dimension)
    #   for m in M
    #     for k in K
    #         parfor s in (tileSize / bankWidth)
    #             addr = k * tile_size + m * K * tile_size + s * bankWidth
    #  Tile size = array_width / 8
    data_str += [format_scalar_definition("int32_t", "Atlbound0", K)]
    data_str += [format_scalar_definition("int32_t", "Atlstride0", a_array_width / 8)]
    data_str += [format_scalar_definition("int32_t", "Atlbound1", M)]
    data_str += [format_scalar_definition("int32_t", "Atlstride1", K * a_array_width / 8)]
    data_str += [format_scalar_definition("int32_t", "Atlbound2", N)]
    data_str += [format_scalar_definition("int32_t", "Atlstride2", 0)]
    data_str += [format_scalar_definition("int32_t", "Atlbound3", 1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride3", 0)]

    # TODO [RG] what does this mean exactly?
    A_enabled_channel_CSR_num = int(math.ceil(a_array_width / BANKWIDTH / 32))
    channel_en_A = [0] * A_enabled_channel_CSR_num
    # Related to if this is a wide channel or not. If wide, must be divisible by 8, if narrow, must be divisible by 1
    channel_en_A_bits = max(8, int((Mu * Ku * nbit_a / BANKWIDTH + 7) // 8 * 8))
    channel_en_A = gen_channel_enable_CSR(channel_en_A, channel_en_A_bits)
    data_str += ["int32_t channel_en_A[] = { " + ", ".join(map(str, channel_en_A)) + " };"]

    # -------------------
    # B streamer setting
    # -------------------
    data_str += [format_scalar_definition("int32_t", "Bslstride0", BANKWIDTH / 8)]

    data_str += [format_scalar_definition("int32_t", "Btlbound0", K)]
    data_str += [format_scalar_definition("int32_t", "Btlstride0", b_array_width / 8)]
    data_str += [format_scalar_definition("int32_t", "Btlbound1", M)]
    data_str += [format_scalar_definition("int32_t", "Btlstride1", K * b_array_width / 8)]
    data_str += [format_scalar_definition("int32_t", "Btlbound2", N)]
    data_str += [format_scalar_definition("int32_t", "Btlstride2", 0)]
    data_str += [format_scalar_definition("int32_t", "Btlbound3", 1)]
    data_str += [format_scalar_definition("int32_t", "Btlstride3", 0)]

    B_enabled_channel_CSR_num = int(math.ceil(b_array_width / BANKWIDTH / 32))
    channel_en_B = [0] * B_enabled_channel_CSR_num
    channel_en_B_bits = max(8, int((Nu * Ku * nbit_b / BANKWIDTH + 7) // 8 * 8))
    channel_en_B = gen_channel_enable_CSR(channel_en_B, channel_en_B_bits)
    data_str += ["int32_t channel_en_B[] = { " + ", ".join(map(str, channel_en_B)) + " };"]

    # -------------------
    # C streamer setting
    # -------------------
    # NOTE [RG] we don't have C for now

    # spatial settings
    data_str += [format_scalar_definition("int32_t", "Cslstride0", BANKWIDTH / 8)]
    c_spatial_bound_0 = serial_width_d / BANKWIDTH

    # temporal settings # TODO change to order of D
    data_str += [format_scalar_definition("int32_t", "Ctlbound0", max(1, Nu * Mu * nbit_c / serial_width_d))]
    data_str += [format_scalar_definition("int32_t", "Ctlstride0", c_spatial_bound_0 * (BANKWIDTH / 8))]
    data_str += [format_scalar_definition("int32_t", "Ctlbound1", N)]
    data_str += [format_scalar_definition("int32_t", "Ctlstride1", nbit_c * Mu * Nu / 8)]
    data_str += [format_scalar_definition("int32_t", "Ctlbound2", M)]
    data_str += [format_scalar_definition("int32_t", "Ctlstride2", N * nbit_c * Mu * Nu / 8)]
    data_str += [format_scalar_definition("int32_t", "Ctlbound3", 1)]
    data_str += [format_scalar_definition("int32_t", "Ctlstride3", 0)]

    disable_C = 1
    enable_full_C = 0
    assert disable_C or enable_full_C, "Invalid C settings"

    C_enabled_channel_CSR_num = int(math.ceil(serial_width_d / BANKWIDTH / 32))
    channel_en_C = [0] * C_enabled_channel_CSR_num

    if enable_full_C == 1:
        channel_en_C_bits = int((Mu * Nu * nbit_c / BANKWIDTH + 7) // 8 * 8)
    else:
        channel_en_C_bits = 0

    channel_en_C = gen_channel_enable_CSR(channel_en_C, channel_en_C_bits)
    data_str += ["int32_t channel_en_C[] = { " + ", ".join(map(str, channel_en_C)) + " };"]

    # -------------------
    # D streamer setting
    # -------------------
    # spatial settings
    data_str += [format_scalar_definition("int32_t", "Dslstride0", BANKWIDTH / 8)]

    # Temporal settings. Order: N, M, K
    assert d_array_width % serial_width_d == 0, "d_array_width must be divisible by serial_width_d"
    data_str += [format_scalar_definition("int32_t", "Dtlbound0", d_array_width / serial_width_d)]
    data_str += [format_scalar_definition("int32_t", "Dtlstride0", serial_width_d / 8)]
    data_str += [format_scalar_definition("int32_t", "Dtlbound1", M)]
    data_str += [format_scalar_definition("int32_t", "Dtlstride1", d_array_width / 8)]
    data_str += [format_scalar_definition("int32_t", "Dtlbound2", N)]
    data_str += [format_scalar_definition("int32_t", "Dtlstride2", M * d_array_width / 8)]
    data_str += [format_scalar_definition("int32_t", "Dtlbound3", 1)]
    data_str += [format_scalar_definition("int32_t", "Dtlstride3", 0)]

    D_enabled_channel_CSR_num = int(math.ceil(serial_width_d / BANKWIDTH / 32))

    channel_en_D = [0] * D_enabled_channel_CSR_num
    channel_en_D_bits = int((Mu * Nu * nbit_c / BANKWIDTH + 7) // 8 * 8)
    channel_en_D = gen_channel_enable_CSR(channel_en_D, channel_en_D_bits)
    data_str += ["int32_t channel_en_D[] = { " + ", ".join(map(str, channel_en_D)) + " };"]

    # ------------
    # Base address
    # ------------

    # Start address of the data (relative to some base address)
    delta_local_a = 0
    delta_local_b = align_wide_addr(delta_local_a + data_length_a)
    delta_local_c = align_wide_addr(delta_local_b + data_length_b)
    delta_local_d = align_wide_addr(delta_local_c + data_length_c)
    data_str += [format_scalar_definition("int32_t", "delta_local_a", delta_local_a)]
    data_str += [format_scalar_definition("int32_t", "delta_local_b", delta_local_b)]
    data_str += [format_scalar_definition("int32_t", "delta_local_c", delta_local_c)]
    data_str += [format_scalar_definition("int32_t", "delta_local_d", delta_local_d)]

    # -------------------
    # Test Data generation
    # -------------------

    # Parse test data from external file.
    # TODO these file names are hardcoded too much
    try:
        A_int = read_data_int("A.bin")
        B_int = read_data_int("B.bin")
        C_int = read_data_int("C.bin")
        D_int = read_data_int("D.bin")
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Error loading test data: {e}. Did you run the scala data generator and is the data directory correct?"
        )

    data_str += [format_vector_definition("uint16_t", "A", A_int)]
    data_str += [format_vector_definition("uint16_t", "B", B_int)]
    data_str += [format_vector_definition("uint16_t", "C", C_int)]
    data_str += [format_vector_definition("uint16_t", "D", D_int)]

    # NOTE [RG] why do whe need this
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_A", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_B", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_C", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_D", 0)]

    data_str = "\n\n".join(data_str)

    return data_str


def read_data_int(filename):
    """Read a vec from a file."""
    with open(os.path.join(DATA_OUT_DIR, filename), "r") as f:
        lines = f.readlines()
    data_lines = [line.strip() for line in lines if not line.startswith("#")]
    return [int(x) for x in data_lines]


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
    print(emit_header_file(**merged_config))


if __name__ == "__main__":
    main()
