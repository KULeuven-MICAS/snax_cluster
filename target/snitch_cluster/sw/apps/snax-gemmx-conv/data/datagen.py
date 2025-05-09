#!/usr/bin/env python3

# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

import numpy as np
import argparse
import pathlib
import hjson
import sys
import os
import math

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402

# Add golden model path
from snax_utils import (  # noqa E402
    conv2d,
    im2col,
    postprocessing_simd_golden_model,
    align_wide_addr,
)  # noqa E402

np.random.seed(42)


# Add stdint.h header
def emit_header_file(**kwargs):
    emit_str = "#include <stdint.h>\n\n"
    emit_str += emit_gemmx_data(**kwargs)
    return emit_str


MIN = -128
MAX = 127
MIN_BIAS = -(2**30)
MAX_BIAS = 2**30 - 1

bankWidth = 64
input_data_width = 8
output_data_width = 32
quantized_output_data_width = 8


def emit_conv_data(**kwargs):

    meshRow = kwargs["snax_streamer_gemmX_core_template"]["snax_acc_cfg"][0][
        "snax_gemmx_mesh_row"
    ]
    tileSize = kwargs["snax_streamer_gemmX_core_template"]["snax_acc_cfg"][0][
        "snax_gemmx_tile_size"
    ]
    meshCol = kwargs["snax_streamer_gemmX_core_template"]["snax_acc_cfg"][0][
        "snax_gemmx_mesh_col"
    ]
    snax_gemmx_serial_c32_d32_width = kwargs["snax_streamer_gemmX_core_template"][
        "snax_acc_cfg"
    ][0]["snax_gemmx_serial_c32_d32_width"]
    snax_gemmx_serial_d8_width = kwargs["snax_streamer_gemmX_core_template"][
        "snax_acc_cfg"
    ][0]["snax_gemmx_serial_d8_width"]

    # size extraction
    Cin = kwargs["Cin"]
    Cout = kwargs["Cout"]

    Nbatch, CinTemp, H, W = (
        kwargs["Nbatch"],
        kwargs["Cin"] // tileSize,
        kwargs["H"],
        kwargs["W"],
    )
    CoutTemp, Kh, Kw = (
        kwargs["Cout"] // meshCol,
        kwargs["Kh"],
        kwargs["Kw"],
    )

    stride_h, stride_w = (kwargs["stride_h"], kwargs["stride_w"])
    pad_h, pad_w = (kwargs["pad_h"], kwargs["pad_w"])

    # make sure the output width is multiple of meshRow
    if W // stride_w % meshRow != 0:
        W = W + (stride_w * (meshRow - (W // stride_w) % meshRow)) % (
            stride_w * meshRow
        )

    # test data generation
    input_data = np.random.randint(-10, 10, size=(Nbatch, CinTemp, H, W, tileSize))
    kernel = np.random.randint(
        -10, 10, size=(CoutTemp, CinTemp, Kh, Kw, meshCol, tileSize)
    )

    # inferred config from the input data and kernel
    padding = pad_h, pad_w
    stride = stride_h, stride_w

    # Padding the input data
    input_padding = np.pad(
        input_data,
        ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode="constant",
    )

    # Calculate the size of the output feature map
    out_width = (W + 2 * pad_w - Kw) // stride_w + 1
    out_height = (H + 2 * pad_h - Kh) // stride_h + 1

    assert out_width % meshRow == 0, "out_width must be multiple of meshRow"

    M = out_height * out_width // meshRow
    K = Cin // tileSize * Kh * Kw
    N = Cout // meshCol

    length_c = M * N * meshRow * meshCol

    enabled_channel_CSR_num = int(
        math.ceil((meshRow * meshCol) * output_data_width / bankWidth / 32)
    )

    broadcast_C = kwargs["broadcast_C"] == 1 and kwargs["channel_en_C"] == 1
    disable_C = kwargs["broadcast_C"] == 0 and kwargs["channel_en_C"] == 0
    enable_full_C = kwargs["broadcast_C"] == 0 and kwargs["channel_en_C"] == 1

    assert broadcast_C or disable_C or enable_full_C, "Invalid C settings"
    if broadcast_C == 1:
        bias = np.random.randint(
            MIN_BIAS, MAX_BIAS, size=(int(length_c / meshRow / meshCol), 1, meshCol)
        )
        bias = np.repeat(bias, repeats=meshRow, axis=1).reshape(-1)
    elif enable_full_C == 1:
        bias = np.random.randint(MIN_BIAS, MAX_BIAS, size=length_c).reshape(-1)
    else:
        bias = np.random.randint(0, 1, size=length_c).reshape(-1)

    data_str = []

    data_str += [
        format_scalar_definition("int32_t", "broadcast_C", kwargs["broadcast_C"])
    ]

    if broadcast_C == 1:
        assert meshCol * output_data_width % bankWidth == 0
        # Note: if C is hanged to wide ports, the number of bits to enable is
        # multipliers of 8 (8 narrow channels equal to 1 wide channel)
        channel_en_C_1_bits = int(
            (meshCol * output_data_width / bankWidth + 7) // 8 * 8
        )
        # Generate the elements
        channel_en_C = [0] * enabled_channel_CSR_num  # Initialize with zeros

        for i in range(channel_en_C_1_bits):
            element_index = i // 32  # Determine which element to modify
            bit_position = i % 32  # Position within the element
            if element_index < enabled_channel_CSR_num:
                channel_en_C[element_index] |= 1 << (bit_position)

        # Convert elements to integers
        channel_en_C = [int(x) for x in channel_en_C][::-1]  # Reverse the list
    elif enable_full_C == 1:
        channel_en_C = [((1 << 32) - 1) for i in range(enabled_channel_CSR_num)]
    else:
        channel_en_C = [0 for i in range(enabled_channel_CSR_num)]
    data_str += [
        "int32_t channel_en_C[] = { " + ", ".join(map(str, channel_en_C)) + " };"
    ]

    # Generating conv2d settings
    data_str += [
        format_scalar_definition("int", "Nbatch", Nbatch),
        format_scalar_definition("int", "H", H),
        format_scalar_definition("int", "W", W),
        format_scalar_definition("int", "Cin", Cin),
        format_scalar_definition("int", "Cout", Cout),
        format_scalar_definition("int", "Kh", Kh),
        format_scalar_definition("int", "Kw", Kw),
        format_scalar_definition("int", "stride_h", stride_h),
        format_scalar_definition("int", "stride_w", stride_w),
        format_scalar_definition("int", "pad_h", pad_h),
        format_scalar_definition("int", "pad_w", pad_w),
    ]

    # Generating matrix size settings
    data_str += [
        format_scalar_definition("int", "Batch", Nbatch),
        format_scalar_definition("int", "M", M),
        format_scalar_definition("int", "K", K),
        format_scalar_definition("int", "N", N),
    ]

    # Generating base pointer settings

    if kwargs["interleaved_address"] == 1:
        # Generating base pointer settings, interleaved memory
        delta_local_a = 0

        delta_local_b = input_padding.size
        assert (
            input_padding.size
            == Nbatch * CinTemp * (H + 2 * pad_h) * (W + 2 * pad_w) * tileSize
        )

        delta_local_b = align_wide_addr(delta_local_b, 64)
        assert delta_local_b % 64 == 0

        delta_local_c = delta_local_b + kernel.size
        assert kernel.size == CoutTemp * CinTemp * Kh * Kw * tileSize * meshCol
        delta_local_c = align_wide_addr(delta_local_c, 64)
        assert delta_local_c % 64 == 0

        delta_local_d8 = delta_local_c + length_c * 4
        delta_local_d8 = align_wide_addr(delta_local_d8, 64)
        assert delta_local_d8 % 64 == 0

        delta_local_d32 = delta_local_d8

        # logical address is the same as physical address
        delta_physical_a = delta_local_a
        delta_physical_b = delta_local_b
        delta_physical_c = delta_local_c
        delta_physical_d8 = delta_local_d8
        delta_physical_d32 = delta_local_d32

        assert (
            input_padding.size + kernel.size + length_c * 4 * 2
            < kwargs["memory_size"] * 1024
        )
    else:
        # Generating base pointer settings
        base_logical_addr_delta = kwargs["memory_size"] / 4 * 1024
        delta_local_a = 0
        delta_local_b = base_logical_addr_delta
        delta_local_c = base_logical_addr_delta * 2
        delta_local_d32 = base_logical_addr_delta * 3
        delta_local_d8 = base_logical_addr_delta * 3

        base_pyhsical_addr_delta = 64
        delta_physical_a = 0
        delta_physical_b = base_pyhsical_addr_delta
        delta_physical_c = base_pyhsical_addr_delta * 2
        delta_physical_d32 = base_pyhsical_addr_delta * 3
        delta_physical_d8 = base_pyhsical_addr_delta * 3

        assert (
            input_padding.size < base_logical_addr_delta
            and kernel.size < base_logical_addr_delta
            and M * N * meshRow * meshCol * (output_data_width / 8)
            < base_logical_addr_delta
        )

    if kwargs["interleaved_address"] == 1:
        # logical address is the same as physical address
        data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_A", 0)]
        data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_B", 0)]
        data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_C", 0)]
        data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_D32", 0)]
        data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_D8", 0)]
    else:
        # open the address remap
        data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_A", 2)]
        data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_B", 2)]
        data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_C", 2)]
        data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_D32", 2)]
        data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_D8", 2)]

    data_str += [
        format_scalar_definition(
            "int32_t", "interleaved_address", kwargs["interleaved_address"]
        )
    ]

    data_str += [
        format_scalar_definition("int32_t", "delta_physical_a", delta_physical_a),
        format_scalar_definition("int32_t", "delta_physical_b", delta_physical_b),
        format_scalar_definition("int32_t", "delta_physical_c", delta_physical_c),
        format_scalar_definition("int32_t", "delta_physical_d8", delta_physical_d8),
        format_scalar_definition("int32_t", "delta_physical_d32", delta_physical_d32),
    ]

    data_str += [
        format_scalar_definition("int32_t", "delta_local_a", delta_local_a),
        format_scalar_definition("int32_t", "delta_local_b", delta_local_b),
        format_scalar_definition("int32_t", "delta_local_c", delta_local_c),
        format_scalar_definition("int32_t", "delta_local_d8", delta_local_d8),
        format_scalar_definition("int32_t", "delta_local_d32", delta_local_d32),
    ]

    # for streamer cfg
    # -----------------------------------------------------------
    # streamer setting for data mover A
    # -----------------------------------------------------------
    # NC8HW8
    # Aslstride0 = tileSize * stride_w
    Aslstride0 = 8

    # K dim
    Atlbound0 = Kw
    Atlstride0 = tileSize

    Atlbound1 = Kh
    Atlstride1 = tileSize * (W + 2 * pad_w)

    Atlbound2 = CinTemp
    Atlstride2 = tileSize * (W + 2 * pad_w) * (H + 2 * pad_h)

    # N dim
    Atlbound3 = Cout // meshCol
    Atlstride3 = 0

    # M dim
    Atlbound4 = out_width // meshRow
    Atlstride4 = meshRow * tileSize * stride_w

    Atlbound5 = out_height
    Atlstride5 = tileSize * (W + 2 * pad_w) * stride_h

    assert (
        Atlstride0 % (bankWidth / 8) == 0
        and Atlstride1 % (bankWidth / 8) == 0
        and Atlstride2 % (bankWidth / 8) == 0
        and Atlstride3 % (bankWidth / 8) == 0
        and Atlstride4 % (bankWidth / 8) == 0
        and Atlstride5 % (bankWidth / 8) == 0
    )

    assert (
        M * K * N
        == Atlbound0 * Atlbound1 * Atlbound2 * Atlbound3 * Atlbound4 * Atlbound5
    )

    data_str += [
        format_scalar_definition("int32_t", "Aslstride0", Aslstride0),
        format_scalar_definition("int32_t", "Atlbound0", Atlbound0),
        format_scalar_definition("int32_t", "Atlstride0", Atlstride0),
        format_scalar_definition("int32_t", "Atlbound1", Atlbound1),
        format_scalar_definition("int32_t", "Atlstride1", Atlstride1),
        format_scalar_definition("int32_t", "Atlbound2", Atlbound2),
        format_scalar_definition("int32_t", "Atlstride2", Atlstride2),
        format_scalar_definition("int32_t", "Atlbound3", Atlbound3),
        format_scalar_definition("int32_t", "Atlstride3", Atlstride3),
        format_scalar_definition("int32_t", "Atlbound4", Atlbound4),
        format_scalar_definition("int32_t", "Atlstride4", Atlstride4),
        format_scalar_definition("int32_t", "Atlbound5", Atlbound5),
        format_scalar_definition("int32_t", "Atlstride5", Atlstride5),
    ]

    # CoutTempCinTempFyFx88
    # -----------------------------------------------------------
    # streamer setting for data mover B
    # -----------------------------------------------------------
    Bslstride0 = bankWidth / 8

    # K dim
    Btlbound0 = Kw * Kh * CinTemp
    Btlstride0 = tileSize * meshCol

    # N dim
    Btlbound1 = Cout // meshCol
    Btlstride1 = tileSize * meshCol * Kw * Kh * CinTemp

    # M dim
    Btlbound2 = out_width * out_height // meshRow
    Btlstride2 = 0

    assert (
        Btlstride0 % (bankWidth / 8) == 0
        and Btlstride1 % (bankWidth / 8) == 0
        and Btlstride2 % (bankWidth / 8) == 0
    )

    assert K * N * M == Btlbound0 * Btlbound1 * Btlbound2, (
        "K * N * M",
        K * N * M,
        "Loopbounds multipliers ",
        Btlbound0 * Btlbound1 * Btlbound2,
    )

    data_str += [
        format_scalar_definition("int32_t", "Bslstride0", Bslstride0),
        format_scalar_definition("int32_t", "Btlbound0", Btlbound0),
        format_scalar_definition("int32_t", "Btlstride0", Btlstride0),
        format_scalar_definition("int32_t", "Btlbound1", Btlbound1),
        format_scalar_definition("int32_t", "Btlstride1", Btlstride1),
        format_scalar_definition("int32_t", "Btlbound2", Btlbound2),
        format_scalar_definition("int32_t", "Btlstride2", Btlstride2),
    ]

    # -----------------------------------------------------------
    # streamer setting for data mover C
    # -----------------------------------------------------------
    # C is int32_t so the stride is 4 times of the int8_t
    # NHWC
    Cslstride0 = bankWidth / 8

    # serial input
    c32_spatial_bound_0 = snax_gemmx_serial_c32_d32_width // bankWidth
    Ctlbound0 = output_data_width * meshRow * meshCol / snax_gemmx_serial_c32_d32_width
    Ctlstride0 = (bankWidth / 8) * c32_spatial_bound_0

    # N dim
    Ctlbound1 = Cout // meshCol
    Ctlstride1 = out_height * out_width * meshCol * (output_data_width // 8)

    # M dim
    # K is merged because of the block gemm output stationarity
    Ctlbound2 = out_width // meshRow
    Ctlstride2 = meshRow * meshCol * (output_data_width // 8)

    Ctlbound3 = out_height
    Ctlstride3 = out_width * meshCol * (output_data_width // 8)

    assert (
        Ctlstride0 % (bankWidth / 8) == 0
        and Ctlstride1 % (bankWidth / 8) == 0
        and Ctlstride2 % (bankWidth / 8) == 0
        and Ctlstride3 % (bankWidth / 8) == 0
    )

    data_str += [
        format_scalar_definition("int32_t", "Cslstride0", Cslstride0),
        format_scalar_definition("int32_t", "Ctlbound0", Ctlbound0),
        format_scalar_definition("int32_t", "Ctlstride0", Ctlstride0),
        format_scalar_definition("int32_t", "Ctlbound1", Ctlbound1),
        format_scalar_definition("int32_t", "Ctlstride1", Ctlstride1),
        format_scalar_definition("int32_t", "Ctlbound2", Ctlbound2),
        format_scalar_definition("int32_t", "Ctlstride2", Ctlstride2),
        format_scalar_definition("int32_t", "Ctlbound3", Ctlbound3),
        format_scalar_definition("int32_t", "Ctlstride3", Ctlstride3),
    ]

    # -----------------------------------------------------------
    # streamer setting for data mover D32
    # -----------------------------------------------------------
    D32slstride0 = bankWidth / 8

    # serial output
    d32_spatial_bound_0 = snax_gemmx_serial_c32_d32_width // bankWidth
    D32tlbound0 = (
        output_data_width * meshRow * meshCol / snax_gemmx_serial_c32_d32_width
    )
    D32tlstride0 = (bankWidth / 8) * d32_spatial_bound_0

    # N dim
    D32tlbound1 = Cout // meshCol
    D32tlstride1 = out_height * out_width * meshCol * (output_data_width // 8)

    # M dim
    # K is merged because of the block gemm output stationarity
    D32tlbound2 = out_width // meshRow
    D32tlstride2 = meshRow * meshCol * (output_data_width // 8)

    D32tlbound3 = out_height
    D32tlstride3 = out_width * meshCol * (output_data_width // 8)

    assert (
        D32tlstride0 % (bankWidth / 8) == 0
        and D32tlstride1 % (bankWidth / 8) == 0
        and D32tlstride2 % (bankWidth / 8) == 0
        and D32tlstride3 % (bankWidth / 8) == 0
    )

    data_str += [
        format_scalar_definition("int32_t", "D32slstride0", D32slstride0),
        format_scalar_definition("int32_t", "D32tlbound0", D32tlbound0),
        format_scalar_definition("int32_t", "D32tlstride0", D32tlstride0),
        format_scalar_definition("int32_t", "D32tlbound1", D32tlbound1),
        format_scalar_definition("int32_t", "D32tlstride1", D32tlstride1),
        format_scalar_definition("int32_t", "D32tlbound2", D32tlbound2),
        format_scalar_definition("int32_t", "D32tlstride2", D32tlstride2),
        format_scalar_definition("int32_t", "D32tlbound3", D32tlbound3),
        format_scalar_definition("int32_t", "D32tlstride3", D32tlstride3),
    ]

    # -----------------------------------------------------------
    # streamer setting for data mover D8
    # -----------------------------------------------------------
    # postprocessing D8 settings
    D8slstride0 = bankWidth / 8

    # serial output
    d8_spatial_bound_0 = snax_gemmx_serial_d8_width // bankWidth
    D8tlbound0 = (
        quantized_output_data_width * meshRow * meshCol / snax_gemmx_serial_d8_width
    )
    D8tlstride0 = (bankWidth / 8) * d8_spatial_bound_0

    # N dim
    D8tlbound1 = Cout // meshCol
    D8tlstride1 = out_height * out_width * meshCol * (quantized_output_data_width // 8)

    # M dim
    # K is merged because of the block gemm output stationarity
    D8tlbound2 = out_width // meshRow
    D8tlstride2 = meshRow * meshCol * (quantized_output_data_width // 8)

    D8tlbound3 = out_height
    D8tlstride3 = out_width * meshCol * (quantized_output_data_width // 8)

    assert (
        D8tlstride0 % (bankWidth / 8) == 0
        and D8tlstride1 % (bankWidth / 8) == 0
        and D8tlstride2 % (bankWidth / 8) == 0
        and D8tlstride3 % (bankWidth / 8) == 0
    )

    data_str += [
        format_scalar_definition("int32_t", "D8slstride0", D8slstride0),
        format_scalar_definition("int32_t", "D8tlbound0", D8tlbound0),
        format_scalar_definition("int32_t", "D8tlstride0", D8tlstride0),
        format_scalar_definition("int32_t", "D8tlbound1", D8tlbound1),
        format_scalar_definition("int32_t", "D8tlstride1", D8tlstride1),
        format_scalar_definition("int32_t", "D8tlbound2", D8tlbound2),
        format_scalar_definition("int32_t", "D8tlstride2", D8tlstride2),
        format_scalar_definition("int32_t", "D8tlbound3", D8tlbound3),
        format_scalar_definition("int32_t", "D8tlstride3", D8tlstride3),
    ]

    # Generating set integer a and b for subtraction to 0
    subtraction_a = 0
    subtraction_b = 0

    # Writing the subtraction value to data.h
    data_str += [
        format_scalar_definition("int8_t", "subtraction_a", subtraction_a),
        format_scalar_definition("int8_t", "subtraction_b", subtraction_b),
    ]

    # direct conv2d
    direct_conv2d_res = conv2d(
        input_data,
        kernel,
        stride=stride,
        padding=padding,
        mode="C8HW8",
        hw_sizes={"meshRow": meshRow, "meshCol": meshCol, "tileSize": tileSize},
    )

    # output in NHWC format
    direct_conv2d_res = np.add(direct_conv2d_res.reshape(-1), bias)

    # Writing testing data and golden data into data.h
    # implicit im2col matrix and kernel, store original input data and kernel
    data_str += [format_vector_definition("int8_t", "A", input_padding.reshape(-1))]
    data_str += [format_vector_definition("int8_t", "B", kernel.reshape(-1))]
    data_str += [format_vector_definition("int32_t", "C", bias.reshape(-1))]

    data_str += [format_scalar_definition("int32_t", "transposed_A", 0)]
    data_str += [format_scalar_definition("int32_t", "transposed_B", 0)]

    return data_str, direct_conv2d_res


def emit_gemmx_data(**kwargs):

    data_str, D32 = emit_conv_data(**kwargs)

    data_str += [format_vector_definition("int32_t", "D32", D32)]

    # -----------------------------------------------------------
    # Postprocessing
    # -----------------------------------------------------------

    bypassSIMD = kwargs["bypassSIMD"]
    data_str += [format_scalar_definition("int32_t", "bypassSIMD", bypassSIMD)]

    # Generating random constant values
    group_num = kwargs["snax_streamer_gemmX_core_template"]["snax_acc_cfg"][0][
        "snax_gemmx_mesh_col"
    ]

    input_zp_i = np.random.randint(MIN, MAX)
    output_zp_i = np.random.randint(MIN, MAX)
    max_int_i = MAX
    min_int_i = MIN
    double_round_i = np.random.randint(0, 1)

    shift_i = np.random.randint(0, 63, size=group_num)  # values between 0-63
    multiplier_i = np.random.randint(-(2**31), 2**31 - 1, size=group_num)

    # Writing the constant values to data.h
    data_str += [
        format_scalar_definition("int8_t", "input_zp_i", input_zp_i),
        format_scalar_definition("int8_t", "output_zp_i", output_zp_i),
        format_scalar_definition("int8_t", "max_int_i", max_int_i),
        format_scalar_definition("int8_t", "min_int_i", min_int_i),
        format_scalar_definition("int8_t", "double_round_i", double_round_i),
    ]

    shared_bitpacked_shift_i = [
        (shift_i[i + 3] << 24)
        | (shift_i[i + 2] << 16)
        | (shift_i[i + 1] << 8)
        | shift_i[i]
        for i in range(0, group_num, 4)
    ]

    data_str += [
        (
            "int32_t shared_bitpacked_shift[] = { "
            + ", ".join(map(str, shared_bitpacked_shift_i))
            + " };"
        )
    ]
    data_str += [
        "int32_t shared_multiplier[] = { " + ", ".join(map(str, multiplier_i)) + " };"
    ]

    D8 = np.zeros_like(D32, dtype=np.uint8)
    # output channel (innermost dim) has a different scale factor
    for i in range(group_num):
        D8[i::group_num] = postprocessing_simd_golden_model(
            D32[i::group_num],
            input_zp_i,
            output_zp_i,
            shift_i[i],
            max_int_i,
            min_int_i,
            double_round_i,
            multiplier_i[i],
        )

    data_str += [format_vector_definition("int8_t", "D8", D8)]

    data_str = "\n\n".join(data_str)

    return data_str


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

    # Load hardware config file
    with args.hwcfg.open() as f:
        hw = hjson.loads(f.read())

    # Merge dictionaries (hw overrides param in case of conflicts)
    merged_config = {**param, **hw}

    # Emit header file
    print(emit_header_file(**merged_config))


if __name__ == "__main__":

    main()
