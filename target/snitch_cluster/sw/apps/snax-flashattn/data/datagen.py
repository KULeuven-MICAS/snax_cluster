#!/usr/bin/env python3
# NOTE: a COPY of snax-gemmx-matmul/data/datagen.py carrying FlashAttention's parameters,
# its tile geometry and its operand bound. The two copies drift silently -- a fix to the
# shared parts has to be made in both. Prefer sharing the generator over copying it again.

# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

import numpy as np
import argparse
import re
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
    block_gemm_golden_model,
    postprocessing_simd_golden_model,
    align_wide_addr,
)  # noqa E402

np.random.seed(42)


def _mesh(kwargs):
    cfg = kwargs["snax_streamer_gemmX_core_template"]["snax_acc_cfg"][0]
    return (
        cfg["snax_gemmx_mesh_row"],
        cfg["snax_gemmx_tile_size"],
        cfg["snax_gemmx_mesh_col"],
    )


def attention_qshift(d):
    """Smallest right shift on the INT8 operands that keeps every score inside FP16.

    A score is a sum of d products of two shifted INT8s, so |S| <= (128>>q)^2 * d.
    Int32ToFp16 saturates past 65504, exp(inf - inf) is NaN, and the softmax invariant
    then fails on exactly the overflowing rows. Real attention scales by 1/sqrt(d) for the
    same reason; bounding the inputs is cheaper here. Deriving q from d lets the head
    dimension move without reintroducing the overflow.
    """
    for q in range(8):
        if (128 >> q) ** 2 * d <= 65504:
            return q
    raise ValueError("no INT8 shift keeps d=%d scores inside FP16" % d)


def emit_geometry_section(**kwargs):
    """The tile geometry as compile-time constants, derived from the shape and the mesh.

    The kernel takes BR/BC/DHEAD/NKV/QSHIFT from here rather than defining its own, so it
    cannot disagree with the descriptors it is handed.
    """
    meshRow, tileSize, meshCol = _mesh(kwargs)
    d = kwargs["K"] * tileSize
    rows = [
        ("BR", kwargs["N"] * meshCol, "query rows     = N*meshCol"),
        ("BC", kwargs["M"] * meshRow, "key columns    = M*meshRow (the tiling knob)"),
        ("DHEAD", d, "head dimension = K*tileSize (a MODEL property, not a knob)"),
        ("NKV", kwargs["NKV"], "KV tiles streamed through the software pipeline"),
        (
            "QSHIFT",
            attention_qshift(d),
            "operand bound, ALREADY APPLIED to A and B below",
        ),
    ]
    w = max(len(str(v)) for _, v, _ in rows)
    body = "\n".join(
        "#define %-6s %*d  // %s" % (name, w, value, note) for name, value, note in rows
    )
    return (
        "// ---- TILE GEOMETRY -------------------------------------------------------"
        "---\n" + body + "\n"
    )


# Add stdint.h header
def emit_header_file(**kwargs):
    emit_str = "#include <stdint.h>\n\n"
    emit_str += emit_geometry_section(**kwargs) + "\n"
    emit_str += emit_gemmx_data(**kwargs)
    return emit_str


MIN = -128
MAX = 127

bankWidth = 64
input_data_width = 8
output_data_width = 32
quantized_output_data_width = 8


def emit_matmul_data(**kwargs):

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

    # matmul settings
    data_str = []

    data_str += [format_scalar_definition("int32_t", "Batch", 1)]
    data_str += [format_scalar_definition("int32_t", "M", kwargs["M"])]
    data_str += [format_scalar_definition("int32_t", "K", kwargs["K"])]
    data_str += [format_scalar_definition("int32_t", "N", kwargs["N"])]

    data_str += [format_scalar_definition("int32_t", "Aslstride0", bankWidth / 8)]
    data_str += [format_scalar_definition("int32_t", "Atlbound0", kwargs["K"])]
    data_str += [
        format_scalar_definition(
            "int32_t", "Atlstride0", input_data_width * tileSize * meshRow / 8
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Atlbound1", kwargs["N"])]
    data_str += [format_scalar_definition("int32_t", "Atlstride1", 0)]
    data_str += [format_scalar_definition("int32_t", "Atlbound2", kwargs["M"])]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "Atlstride2",
            kwargs["K"] * input_data_width * tileSize * meshRow / 8,
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Atlbound3", 1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride3", 0)]
    data_str += [format_scalar_definition("int32_t", "Atlbound4", 1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride4", 0)]
    data_str += [format_scalar_definition("int32_t", "Atlbound5", 1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride5", 0)]

    data_str += [format_scalar_definition("int32_t", "Bslstride0", bankWidth / 8)]
    data_str += [format_scalar_definition("int32_t", "Btlbound0", kwargs["K"])]
    data_str += [
        format_scalar_definition(
            "int32_t", "Btlstride0", input_data_width * tileSize * meshCol / 8
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Btlbound1", kwargs["N"])]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "Btlstride1",
            kwargs["K"] * input_data_width * tileSize * meshCol / 8,
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Btlbound2", kwargs["M"])]
    data_str += [format_scalar_definition("int32_t", "Btlstride2", 0)]

    # -----------------------------------------------------------
    # streamer c32 settings
    # -----------------------------------------------------------
    # spatial settings
    # The C/D32 ports declare spatial_bounds [[8, 4]] -- TWO spatial dimensions, 32
    # channels -- so the streamer wants TWO spatial strides (S_STRIDE_NUM_READER_WRITER_*
    # is 2). Emitting only stride0 made the app pass a 1-element array to a 2-element CSR
    # write, so the second stride came from off the end of a stack array. Per
    # AddressGenUnit, channel i addresses stride0*(i%8) + stride1*(i/8), so 24 of the 32
    # channels took their address from that garbage -- harmless-looking at 8x8, and the
    # out-of-bounds write that corrupted a neighbouring hart's stack at 16x16.
    #
    # The three strides are one consistent progression over a serialised chunk:
    #   spatial 0 : 8 channels x bankWidth/8            =  64 B
    #   spatial 1 : 4 groups   x 64 B                   = 256 B  (one whole chunk)
    #   temporal 0: steps to the NEXT chunk             = 256 B
    # stride1 belongs to the SPATIAL axis, not the temporal loop: assigning it to
    # Ctlstride0 leaves the temporal step 4x short.
    data_str += [format_scalar_definition("int32_t", "Cslstride0", bankWidth / 8)]
    c32_spatial_bound_0 = 8
    data_str += [
        format_scalar_definition(
            "int32_t", "Cslstride1", c32_spatial_bound_0 * (bankWidth / 8)
        )
    ]
    # temporal settings
    # serial input for C
    data_str += [
        format_scalar_definition(
            "int32_t",
            "Ctlbound0",
            output_data_width * meshRow * meshCol / snax_gemmx_serial_c32_d32_width,
        )
    ]
    data_str += [
        format_scalar_definition(
            "int32_t", "Ctlstride0", snax_gemmx_serial_c32_d32_width / 8
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Ctlbound1", kwargs["N"])]
    data_str += [
        format_scalar_definition(
            "int32_t", "Ctlstride1", output_data_width * meshRow * meshCol / 8
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Ctlbound2", kwargs["M"])]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "Ctlstride2",
            kwargs["N"] * output_data_width * meshRow * meshCol / 8,
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Ctlbound3", 1)]
    data_str += [format_scalar_definition("int32_t", "Ctlstride3", 0)]

    # -----------------------------------------------------------
    # streamer d32 settings
    # -----------------------------------------------------------
    # spatial settings
    # The C/D32 ports declare spatial_bounds [[8, 4]] -- TWO spatial dimensions, 32
    # channels -- so the streamer wants TWO spatial strides (S_STRIDE_NUM_READER_WRITER_*
    # is 2). Emitting only stride0 made the app pass a 1-element array to a 2-element CSR
    # write, so the second stride came from off the end of a stack array. Per
    # AddressGenUnit, channel i addresses stride0*(i%8) + stride1*(i/8), so 24 of the 32
    # channels took their address from that garbage -- harmless-looking at 8x8, and the
    # out-of-bounds write that corrupted a neighbouring hart's stack at 16x16.
    #
    # The three strides are one consistent progression over a serialised chunk:
    #   spatial 0 : 8 channels x bankWidth/8            =  64 B
    #   spatial 1 : 4 groups   x 64 B                   = 256 B  (one whole chunk)
    #   temporal 0: steps to the NEXT chunk             = 256 B
    # stride1 belongs to the SPATIAL axis, not the temporal loop: assigning it to
    # Ctlstride0 leaves the temporal step 4x short.
    data_str += [format_scalar_definition("int32_t", "D32slstride0", bankWidth / 8)]
    d32_spatial_bound_0 = 8
    data_str += [
        format_scalar_definition(
            "int32_t", "D32slstride1", d32_spatial_bound_0 * (bankWidth / 8)
        )
    ]
    # temporal settings
    data_str += [
        format_scalar_definition(
            "int32_t",
            "D32tlbound0",
            output_data_width * meshRow * meshCol / snax_gemmx_serial_c32_d32_width,
        )
    ]
    data_str += [
        format_scalar_definition(
            "int32_t", "D32tlstride0", snax_gemmx_serial_c32_d32_width / 8
        )
    ]
    data_str += [format_scalar_definition("int32_t", "D32tlbound1", kwargs["N"])]
    data_str += [
        format_scalar_definition(
            "int32_t", "D32tlstride1", output_data_width * meshRow * meshCol / 8
        )
    ]
    data_str += [format_scalar_definition("int32_t", "D32tlbound2", kwargs["M"])]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "D32tlstride2",
            kwargs["N"] * output_data_width * meshRow * meshCol / 8,
        )
    ]
    data_str += [format_scalar_definition("int32_t", "D32tlbound3", 1)]
    data_str += [format_scalar_definition("int32_t", "D32tlstride3", 0)]

    # -----------------------------------------------------------
    # streamer d8 settings
    # -----------------------------------------------------------

    # spatial settings
    data_str += [format_scalar_definition("int32_t", "D8slstride0", bankWidth / 8)]
    # temporal settings
    d8_spatial_bound_0 = 8
    data_str += [
        format_scalar_definition(
            "int32_t",
            "D8tlbound0",
            quantized_output_data_width
            * meshRow
            * meshCol
            / snax_gemmx_serial_d8_width,
        )
    ]
    data_str += [
        format_scalar_definition(
            "int32_t", "D8tlstride0", d8_spatial_bound_0 * (bankWidth / 8)
        )
    ]
    data_str += [format_scalar_definition("int32_t", "D8tlbound1", kwargs["N"])]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "D8tlstride1",
            quantized_output_data_width * meshRow * meshCol / 8,
        )
    ]
    data_str += [format_scalar_definition("int32_t", "D8tlbound2", kwargs["M"])]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "D8tlstride2",
            kwargs["N"] * quantized_output_data_width * meshRow * meshCol / 8,
        )
    ]
    data_str += [format_scalar_definition("int32_t", "D8tlbound3", 1)]
    data_str += [format_scalar_definition("int32_t", "D8tlstride3", 0)]

    # -----------------------------------------------------------
    delta_local_a = 0
    delta_local_b = (
        kwargs["K"] * kwargs["M"] * (meshRow * tileSize * input_data_width / 8)
    )
    delta_local_b = align_wide_addr(delta_local_b)
    delta_local_c = delta_local_b + kwargs["K"] * kwargs["N"] * (
        meshCol * tileSize * input_data_width / 8
    )
    delta_local_c = align_wide_addr(delta_local_c)
    delta_local_d32 = delta_local_c + kwargs["M"] * kwargs["N"] * (
        meshRow * meshCol * output_data_width / 8
    )
    delta_local_d32 = align_wide_addr(delta_local_d32)
    delta_local_d8 = delta_local_d32
    data_str += [format_scalar_definition("int32_t", "delta_local_a", delta_local_a)]
    data_str += [format_scalar_definition("int32_t", "delta_local_b", delta_local_b)]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "delta_local_c",
            delta_local_c,
        )
    ]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "delta_local_d32",
            delta_local_d32,
        )
    ]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "delta_local_d8",
            delta_local_d8,
        )
    ]

    # -----------------------------------------------------------
    # Test Data generation
    # -----------------------------------------------------------

    # Generating random 8 integer a and b for subtraction
    subtraction_a = np.random.randint(MIN, MAX)
    subtraction_b = np.random.randint(MIN, MAX)

    # Writing the subtraction value to data.h
    data_str += [format_scalar_definition("int8_t", "subtraction_a", subtraction_a)]
    data_str += [format_scalar_definition("int8_t", "subtraction_b", subtraction_b)]

    A = np.random.randint(
        MIN, MAX, size=(kwargs["M"], kwargs["K"], meshRow, tileSize)
    ).reshape(-1)
    B = np.random.randint(
        MIN, MAX, size=(kwargs["K"], kwargs["N"], tileSize, meshCol)
    ).reshape(-1)

    # Bound the operands so the scores stay inside FP16 -- see attention_qshift(). Doing
    # it here rather than in the kernel keeps a shift over every element of A and B off
    # the DM core at run time; numpy's arithmetic shift on int8 is what `>>=` on int8_t
    # is in C, so the data is what the kernel would have produced itself.
    qshift = attention_qshift(kwargs["K"] * tileSize)
    A = A.astype(np.int8) >> qshift
    B = B.astype(np.int8) >> qshift

    data_str += [format_vector_definition("int8_t", "A", A)]
    data_str += [format_vector_definition("int8_t", "B", B)]

    enabled_channel_CSR_num = int(math.ceil(
        (meshRow * meshCol) * output_data_width / bankWidth / 32
    ))

    broadcast_C = kwargs["broadcast_C"] == 1 and kwargs["channel_en_C"] == 1
    disable_C = kwargs["broadcast_C"] == 0 and kwargs["channel_en_C"] == 0
    enable_full_C = kwargs["broadcast_C"] == 0 and kwargs["channel_en_C"] == 1

    assert broadcast_C or disable_C or enable_full_C, "Invalid C settings"

    if broadcast_C == 1:
        C = np.random.randint(MIN, MAX, size=(kwargs["M"], kwargs["N"], 1, meshCol))
        C = np.repeat(C, repeats=meshRow, axis=1).reshape(-1)
    elif enable_full_C == 1:
        C = np.random.randint(
            MIN, MAX, size=(kwargs["M"], kwargs["N"], meshRow, meshCol)
        ).reshape(-1)
    else:
        C = np.random.randint(
            0, 1, size=(kwargs["M"], kwargs["N"], meshRow, meshCol)
        ).reshape(-1)

    if broadcast_C == 1:
        assert meshCol * output_data_width % bankWidth == 0
        # Note: if C is hanged to wide ports, the mimimum number of bits to enable
        # is multipliers of 8 (8 narrow channels equal to 1 wide channel)
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

    data_str += [
        format_scalar_definition("int32_t", "broadcast_C", kwargs["broadcast_C"])
    ]
    data_str += [format_vector_definition("int32_t", "C", C)]

    if kwargs["transposed_A"] == 1:
        A = A.reshape(kwargs["M"], kwargs["K"], meshRow, tileSize)
        A = A.transpose(0, 1, 3, 2).reshape(-1)
    if kwargs["transposed_B"] == 1:
        B = B.reshape(kwargs["K"], kwargs["N"], tileSize, meshCol)
        B = B.transpose(0, 1, 3, 2).reshape(-1)

    data_str += [
        format_scalar_definition("int32_t", "transposed_A", kwargs["transposed_A"])
    ]
    data_str += [
        format_scalar_definition("int32_t", "transposed_B", kwargs["transposed_B"])
    ]

    D32 = block_gemm_golden_model(
        kwargs["M"],
        kwargs["K"],
        kwargs["N"],
        meshRow,
        tileSize,
        meshCol,
        A,
        B,
        subtraction_a,
        subtraction_b,
        C,
    )

    return data_str, D32


def emit_gemmx_data(**kwargs):
    data_str, D32 = emit_matmul_data(**kwargs)

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

    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_A", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_B", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_C", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_D32", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_D8", 0)]

    data_str = "\n\n".join(data_str)

    return data_str



def emit_shape2_section(param, merged_config):
    """The SECOND matmul shape's streamer descriptors, prefixed S2_, appended to data.h.

    FlashAttention's two matmuls stop being the same shape once d != Bc:

        matmul 1  S^T = K.Q^T     M1*meshRow = Bc,  K1*tileSize = d   <- params.hjson
        matmul 2  O^T = V^T.P^T   M2*meshRow = d,   K2*tileSize = Bc  <- here

    M and K swap roles, so every temporal bound and stride changes with them. The second
    shape is DERIVED from the first rather than configured separately -- otherwise the two
    drift apart the moment the tile is retuned:

        M2 = K1 * tileSize / meshRow      (since K1*tileSize = d)
        K2 = M1 * meshRow  / tileSize     (since M1*meshRow  = Bc)

    The descriptors themselves come from the SAME generator as shape 1, with M and K
    swapped, because hand-deriving bounds and strides is where a silent stride error would
    live. Only the scalar descriptors are kept: shape 2 reuses shape 1's buffers, it just
    walks them differently.
    """
    acc = merged_config["snax_streamer_gemmX_core_template"]["snax_acc_cfg"][0]
    mesh_row = acc["snax_gemmx_mesh_row"]
    tile_size = acc["snax_gemmx_tile_size"]

    m1, n1, k1 = int(param["M"]), int(param["N"]), int(param["K"])
    d, bc = k1 * tile_size, m1 * mesh_row
    assert d % mesh_row == 0, f"d={d} is not a multiple of meshRow={mesh_row}"
    assert bc % tile_size == 0, f"Bc={bc} is not a multiple of tileSize={tile_size}"
    m2, n2, k2 = d // mesh_row, n1, bc // tile_size

    shape2 = {**merged_config, "M": m2, "N": n2, "K": k2}
    keep = re.compile(
        r"^int32_t (M|N|K|[ABCD][0-9]*[a-z]*(?:sl|tl)(?:bound|stride)[0-9]+) = "
    )
    lines = [
        "",
        "// ---- SECOND MATMUL SHAPE " + "-" * 52,
        f"// O^T = V^T.P^T : M={m2} N={n2} K={k2}   (d={d}, Bc={bc}, derived from shape 1 above).",
        "// Switched in per dispatch by gemm_set_shape(); only the bounds and strides that",
        "// depend on M and K differ from shape 1.",
    ]
    lines += [
        ln.replace("int32_t ", "static const int32_t S2_", 1)
        for ln in emit_header_file(**shape2).splitlines()
        if keep.match(ln)
    ]
    return "\n".join(lines)


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

    # Emit header file: shape 1's full data set, then shape 2's descriptors appended
    # to the SAME file. One generator, one output, one #include.
    print(emit_header_file(**merged_config))
    print(emit_shape2_section(param, merged_config))


if __name__ == "__main__":

    main()
