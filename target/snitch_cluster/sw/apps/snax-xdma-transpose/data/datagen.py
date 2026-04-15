#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Yunhao Deng <yunhao.deng@kuleuven.be>

import argparse
import os
import pathlib
import re
import sys

import hjson
import numpy as np

# Add data utility path
sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/")
)
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402


MAX_TEMPORAL_DIM = 5
BIT_WIDTH_CONFIG = {
    8: {
        "tile_width": 8,
        "transfer_per_transpose": 1,
        "transposer_csr": 0,
        "dtype": np.dtype("<u1"),
    },
    16: {
        "tile_width": 8,
        "transfer_per_transpose": 2,
        "transposer_csr": 1,
        "dtype": np.dtype("<u2"),
    },
    32: {
        "tile_width": 4,
        "transfer_per_transpose": 1,
        "transposer_csr": 2,
        "dtype": np.dtype("<u4"),
    },
}


def round_up(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def parse_block_layout(layout_name):
    match = re.fullmatch(r"MNM(\d+)N(\d+)", layout_name)
    if match is None:
        raise ValueError(f"Invalid layout: {layout_name}")
    return int(match.group(1)), int(match.group(2))


def validate_block_layout(layout_name, tile_width):
    block_m, block_n = parse_block_layout(layout_name)
    if block_m % tile_width != 0 or block_n % tile_width != 0:
        raise ValueError(
            f"Invalid layout {layout_name}. Block sizes must be multiples of {tile_width}."
        )
    return block_m, block_n


def get_case_config(case_cfg):
    try:
        return BIT_WIDTH_CONFIG[int(case_cfg["BIT_WIDTH"])]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported BIT_WIDTH {case_cfg['BIT_WIDTH']}, expected one of 8, 16, 32"
        ) from exc


def get_padded_shape(case_cfg, tile_width):
    if case_cfg["input_layout"] == "MN":
        return round_up(case_cfg["M"], tile_width), round_up(case_cfg["N"], tile_width)

    block_m, block_n = validate_block_layout(case_cfg["input_layout"], tile_width)
    return round_up(case_cfg["M"], block_m), round_up(case_cfg["N"], block_n)


def flatten_matrix_for_layout(matrix, layout_name, tile_width):
    if layout_name == "MN":
        return matrix.ravel()

    block_m, block_n = validate_block_layout(layout_name, tile_width)
    rows, cols = matrix.shape
    if rows % block_m != 0 or cols % block_n != 0:
        raise ValueError(
            f"Matrix shape {matrix.shape} is incompatible with layout {layout_name}"
        )

    return (
        matrix.reshape(rows // block_m, block_m, cols // block_n, block_n)
        .swapaxes(1, 2)
        .ravel()
    )


def matrix_to_bytes(matrix_vector, dtype):
    return np.frombuffer(np.asarray(matrix_vector, dtype=dtype).tobytes(), dtype=np.uint8)


def pad_temporal_values(values):
    if len(values) > MAX_TEMPORAL_DIM:
        raise ValueError(
            f"Temporal AGU description exceeds {MAX_TEMPORAL_DIM} dimensions: {values}"
        )
    return values + [0] * (MAX_TEMPORAL_DIM - len(values))


def derive_src_agu(rows, cols, layout_name, tile_width, bytes_per_element, transfer_count):
    if layout_name == "MN":
        return {
            "spatial_stride": cols * bytes_per_element,
            "temporal_bounds": [transfer_count, cols // tile_width, rows // tile_width],
            "temporal_strides": [
                8,
                tile_width * bytes_per_element,
                cols * tile_width * bytes_per_element,
            ],
        }

    block_m, block_n = validate_block_layout(layout_name, tile_width)
    if rows % block_m != 0 or cols % block_n != 0:
        raise ValueError(
            f"Input matrix shape {(rows, cols)} is incompatible with layout {layout_name}"
        )

    return {
        "spatial_stride": block_n * bytes_per_element,
        "temporal_bounds": [
            transfer_count,
            block_n // tile_width,
            cols // block_n,
            block_m // tile_width,
            rows // block_m,
        ],
        "temporal_strides": [
            8,
            tile_width * bytes_per_element,
            block_m * block_n * bytes_per_element,
            block_n * tile_width * bytes_per_element,
            cols * block_m * bytes_per_element,
        ],
    }


def derive_dst_agu(
    rows,
    cols,
    layout_name,
    tile_width,
    bytes_per_element,
    transfer_count,
    enable_transpose,
):
    if layout_name == "MN":
        if enable_transpose:
            return {
                "spatial_stride": rows * bytes_per_element,
                "temporal_bounds": [
                    transfer_count,
                    cols // tile_width,
                    rows // tile_width,
                ],
                "temporal_strides": [
                    8,
                    rows * tile_width * bytes_per_element,
                    tile_width * bytes_per_element,
                ],
            }

        return {
            "spatial_stride": cols * bytes_per_element,
            "temporal_bounds": [transfer_count, cols // tile_width, rows // tile_width],
            "temporal_strides": [
                8,
                tile_width * bytes_per_element,
                cols * tile_width * bytes_per_element,
            ],
        }

    block_m, block_n = validate_block_layout(layout_name, tile_width)
    out_rows = cols if enable_transpose else rows
    out_cols = rows if enable_transpose else cols
    if out_rows % block_m != 0 or out_cols % block_n != 0:
        raise ValueError(
            f"Output matrix shape {(out_rows, out_cols)} is incompatible with layout {layout_name}"
        )

    if enable_transpose:
        return {
            "spatial_stride": block_n * bytes_per_element,
            "temporal_bounds": [
                transfer_count,
                block_m // tile_width,
                cols // block_m,
                block_n // tile_width,
                rows // block_n,
            ],
            "temporal_strides": [
                8,
                block_n * tile_width * bytes_per_element,
                block_m * rows * bytes_per_element,
                tile_width * bytes_per_element,
                block_m * block_n * bytes_per_element,
            ],
        }

    return {
        "spatial_stride": block_n * bytes_per_element,
        "temporal_bounds": [
            transfer_count,
            block_n // tile_width,
            cols // block_n,
            block_m // tile_width,
            rows // block_m,
        ],
        "temporal_strides": [
            8,
            tile_width * bytes_per_element,
            block_m * block_n * bytes_per_element,
            block_n * tile_width * bytes_per_element,
            cols * block_m * bytes_per_element,
        ],
    }


def format_c_array(values):
    return "{ " + ", ".join(str(value) for value in values) + " }"


def emit_struct_definition():
    return """typedef struct {
    const char *name;
    uint32_t M;
    uint32_t N;
    uint32_t bit_width;
    uint8_t enable_transpose;
    uint32_t input_bytes;
    uint32_t output_bytes;
    uint32_t spatial_stride_src;
    uint32_t spatial_stride_dst;
    uint32_t temporal_dimension_src;
    uint32_t temporal_dimension_dst;
    uint32_t temporal_bounds_src[5];
    uint32_t temporal_bounds_dst[5];
    uint32_t temporal_strides_src[5];
    uint32_t temporal_strides_dst[5];
    uint32_t transposer_csr[1];
    const uint8_t *input_matrix_bytes;
    const uint8_t *golden_output_bytes;
} transpose_test_case_t;"""


def generate_case(case_cfg, case_idx):
    cfg = get_case_config(case_cfg)
    tile_width = cfg["tile_width"]
    dtype = cfg["dtype"]
    bytes_per_element = dtype.itemsize
    padded_m, padded_n = get_padded_shape(case_cfg, tile_width)

    rng = np.random.default_rng(320 + case_idx)
    matrix_data = np.zeros((padded_m, padded_n), dtype=dtype)
    matrix_data[: case_cfg["M"], : case_cfg["N"]] = rng.integers(
        low=0,
        high=1 << int(case_cfg["BIT_WIDTH"]),
        size=(case_cfg["M"], case_cfg["N"]),
        dtype=dtype,
    )

    input_matrix = flatten_matrix_for_layout(
        matrix_data, case_cfg["input_layout"], tile_width
    )
    output_matrix = matrix_data.T if case_cfg["enable_transpose"] else matrix_data
    output_matrix = flatten_matrix_for_layout(
        output_matrix, case_cfg["output_layout"], tile_width
    )

    input_bytes = matrix_to_bytes(input_matrix, dtype)
    output_bytes = matrix_to_bytes(output_matrix, dtype)
    src_agu = derive_src_agu(
        padded_m,
        padded_n,
        case_cfg["input_layout"],
        tile_width,
        bytes_per_element,
        cfg["transfer_per_transpose"],
    )
    dst_agu = derive_dst_agu(
        padded_m,
        padded_n,
        case_cfg["output_layout"],
        tile_width,
        bytes_per_element,
        cfg["transfer_per_transpose"],
        bool(case_cfg["enable_transpose"]),
    )

    case_name = case_cfg.get("name", f"transpose_case_{case_idx}")
    input_symbol = f"input_matrix_bytes_case_{case_idx}"
    output_symbol = f"golden_output_bytes_case_{case_idx}"

    definitions = [
        format_vector_definition(
            "const uint8_t",
            input_symbol,
            input_bytes,
            alignment=64,
            hex_bits=8,
        ),
        format_vector_definition(
            "const uint8_t",
            output_symbol,
            output_bytes,
            alignment=64,
            hex_bits=8,
        ),
    ]

    initializer = f"""{{
    .name = "{case_name}",
    .M = {int(case_cfg["M"])},
    .N = {int(case_cfg["N"])},
    .bit_width = {int(case_cfg["BIT_WIDTH"])},
    .enable_transpose = {1 if case_cfg["enable_transpose"] else 0},
    .input_bytes = {input_bytes.size},
    .output_bytes = {output_bytes.size},
    .spatial_stride_src = {src_agu["spatial_stride"]},
    .spatial_stride_dst = {dst_agu["spatial_stride"]},
    .temporal_dimension_src = {len(src_agu["temporal_bounds"])},
    .temporal_dimension_dst = {len(dst_agu["temporal_bounds"])},
    .temporal_bounds_src = {format_c_array(pad_temporal_values(src_agu["temporal_bounds"]))},
    .temporal_bounds_dst = {format_c_array(pad_temporal_values(dst_agu["temporal_bounds"]))},
    .temporal_strides_src = {format_c_array(pad_temporal_values(src_agu["temporal_strides"]))},
    .temporal_strides_dst = {format_c_array(pad_temporal_values(dst_agu["temporal_strides"]))},
    .transposer_csr = {{ {cfg["transposer_csr"]} }},
    .input_matrix_bytes = {input_symbol},
    .golden_output_bytes = {output_symbol},
}}"""

    return {
        "definitions": definitions,
        "initializer": initializer,
        "input_bytes": int(input_bytes.size),
        "output_bytes": int(output_bytes.size),
    }


def emit_header_file(config):
    cases = config.get("cases")
    if not isinstance(cases, list) or len(cases) == 0:
        raise ValueError("Configuration must contain a non-empty top-level 'cases' list")

    emit_str = ["#include <stdint.h>", emit_struct_definition()]
    case_initializers = []
    max_case_input_bytes = 0
    max_case_output_bytes = 0

    for case_idx, case_cfg in enumerate(cases):
        generated_case = generate_case(case_cfg, case_idx)
        emit_str.extend(generated_case["definitions"])
        case_initializers.append(generated_case["initializer"])
        max_case_input_bytes = max(
            max_case_input_bytes, generated_case["input_bytes"]
        )
        max_case_output_bytes = max(
            max_case_output_bytes, generated_case["output_bytes"]
        )

    emit_str.append(
        "transpose_test_case_t transpose_test_cases[] = {\n"
        + ",\n".join(case_initializers)
        + "\n};"
    )
    emit_str.append(
        format_scalar_definition(
            "uint32_t", "transpose_test_case_count", len(case_initializers)
        )
    )
    emit_str.append(
        format_scalar_definition(
            "uint32_t", "max_case_input_bytes", max_case_input_bytes
        )
    )
    emit_str.append(
        format_scalar_definition(
            "uint32_t", "max_case_output_bytes", max_case_output_bytes
        )
    )
    return "\n\n".join(emit_str)


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

    with args.cfg.open() as cfg_file:
        config = hjson.loads(cfg_file.read())

    print(emit_header_file(config))


if __name__ == "__main__":
    main()
