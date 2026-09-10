#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Jonas Crols <jonas.crols@student.kuleuven.be>
# Yunhao Deng <yunhao.deng@kuleuven.be>

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402


ELEMENT_WIDTH = 32
ELEMENT_BYTES = ELEMENT_WIDTH // 8
XDMA_WIDTH_BYTES = 64
ELEMENTS_PER_XDMA_BEAT = XDMA_WIDTH_BYTES // ELEMENT_BYTES
RNG_SEED = 320


def round_up(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def emit_header_file(**kwargs):
    emit_str = ["#include <stdint.h>"]
    emit_str += emit_elementwise_add_data(**kwargs)
    return "\n\n".join(emit_str)


def emit_elementwise_add_data(**kwargs):
    element_width = int(kwargs["BIT_WIDTH"])
    if element_width != ELEMENT_WIDTH:
        raise ValueError(
            f"Invalid BIT_WIDTH: {element_width}, only {ELEMENT_WIDTH} is supported"
        )

    rows = int(kwargs["M"])
    cols = int(kwargs["N"])
    if rows <= 0 or cols <= 0:
        raise ValueError(f"M and N must be positive, got M={rows}, N={cols}")

    padded_cols = round_up(cols, ELEMENTS_PER_XDMA_BEAT)
    matrix_shape = (rows, padded_cols)
    matrix_elements = rows * padded_cols
    input_bytes = matrix_elements * ELEMENT_BYTES
    input_bytes_aligned = round_up(input_bytes, XDMA_WIDTH_BYTES)
    output_bytes = input_bytes
    tile_count = matrix_elements // ELEMENTS_PER_XDMA_BEAT

    rng = np.random.default_rng(RNG_SEED)
    int32_info = np.iinfo(np.int32)

    matrix1_data = np.zeros(matrix_shape, dtype=np.int32)
    matrix2_data = np.zeros(matrix_shape, dtype=np.int32)
    matrix1_data[:, :cols] = rng.integers(
        low=int32_info.min,
        high=int32_info.max,
        size=(rows, cols),
        dtype=np.int32,
    )
    matrix2_data[:, :cols] = rng.integers(
        low=int32_info.min,
        high=int32_info.max,
        size=(rows, cols),
        dtype=np.int32,
    )

    golden_output = (
        matrix1_data.view(np.uint32) + matrix2_data.view(np.uint32)
    ).view(np.int32)

    spatial_stride_src = 8
    spatial_stride_dst = 8
    temporal_bounds_src = [2, tile_count]
    temporal_strides_src = [input_bytes_aligned, XDMA_WIDTH_BYTES]
    temporal_bounds_dst = [tile_count]
    temporal_strides_dst = [XDMA_WIDTH_BYTES]

    emit_str = []
    emit_str += [format_scalar_definition("uint32_t", "matrix_m", rows)]
    emit_str += [format_scalar_definition("uint32_t", "matrix_n", cols)]
    emit_str += [format_scalar_definition("uint32_t", "matrix_padded_n", padded_cols)]
    emit_str += [
        format_scalar_definition("uint32_t", "matrix_elements", matrix_elements)
    ]
    emit_str += [format_scalar_definition("uint32_t", "input_bytes", input_bytes)]
    emit_str += [
        format_scalar_definition("uint32_t", "input_bytes_aligned", input_bytes_aligned)
    ]
    emit_str += [format_scalar_definition("uint32_t", "output_bytes", output_bytes)]
    emit_str += [format_scalar_definition("uint32_t", "tile_count", tile_count)]
    emit_str += [
        format_vector_definition(
            "int32_t",
            "input_matrix1",
            matrix1_data.ravel(),
            alignment=XDMA_WIDTH_BYTES,
            hex_bits=ELEMENT_WIDTH,
            cast_hex=True,
        )
    ]
    emit_str += [
        format_vector_definition(
            "int32_t",
            "input_matrix2",
            matrix2_data.ravel(),
            alignment=XDMA_WIDTH_BYTES,
            hex_bits=ELEMENT_WIDTH,
            cast_hex=True,
        )
    ]
    emit_str += [
        format_vector_definition(
            "int32_t",
            "golden_output_matrix",
            golden_output.ravel(),
            alignment=XDMA_WIDTH_BYTES,
            hex_bits=ELEMENT_WIDTH,
            cast_hex=True,
        )
    ]
    emit_str += [
        format_scalar_definition("uint32_t", "spatial_stride_src", spatial_stride_src)
    ]
    emit_str += [
        format_scalar_definition("uint32_t", "spatial_stride_dst", spatial_stride_dst)
    ]
    emit_str += [
        format_vector_definition("uint32_t", "temporal_bounds_src", temporal_bounds_src)
    ]
    emit_str += [
        format_vector_definition("uint32_t", "temporal_bounds_dst", temporal_bounds_dst)
    ]
    emit_str += [
        format_vector_definition(
            "uint32_t", "temporal_strides_src", temporal_strides_src
        )
    ]
    emit_str += [
        format_vector_definition(
            "uint32_t", "temporal_strides_dst", temporal_strides_dst
        )
    ]
    emit_str += [
        format_scalar_definition(
            "uint32_t", "temporal_dimension_src", len(temporal_bounds_src)
        )
    ]
    emit_str += [
        format_scalar_definition(
            "uint32_t", "temporal_dimension_dst", len(temporal_bounds_dst)
        )
    ]

    return emit_str


def main():
    parser = argparse.ArgumentParser(description="Generating data for kernels")
    parser.add_argument(
        "-c",
        "--cfg",
        type=pathlib.Path,
        required=True,
        help="Select param config file kernel",
    )
    args = parser.parse_args()

    with args.cfg.open() as f:
        param = hjson.loads(f.read())

    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
