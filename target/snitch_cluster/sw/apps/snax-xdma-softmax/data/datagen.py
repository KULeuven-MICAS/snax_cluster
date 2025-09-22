#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Jonas Crols <jonas.crols@student.kuleuven.be>

import argparse
import math
import os
import pathlib
import sys

import debugpy

import hjson
import numpy as np

# Add data utility path
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../../../../../../util/sim/",
    )
)
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402
from snax_utils import integer_softmax  # noqa E402


np.random.seed(320)


# Add stdint.h header
def emit_header_file(**kwargs):
    emit_str = ["#include <stdint.h>"]
    emit_str += emit_softmax_data(**kwargs)
    return "\n\n".join(emit_str)


def emit_softmax_data(**kwargs):

    data_in_type = "int32_t"
    data_out_type = "uint32_t"

    emit_str = []
    M = kwargs["M"]
    channel_count = kwargs["channels"]


    # Scaling Matrix for softmax
    matrix_scaling = np.zeros((5, channel_count), dtype=np.int32)
    for i in range(channel_count):
        matrix_scaling[0, i] = int(math.log(2) * kwargs["inverse_scaling_factor"])
        matrix_scaling[1, i] = int(0.3585 * kwargs["inverse_scaling_factor"])
        matrix_scaling[2, i] = int(1.353 * kwargs["inverse_scaling_factor"])
        matrix_scaling[3, i] = int(0.344 * (kwargs["inverse_scaling_factor"] ** 3)) >> math.floor(math.log2(kwargs["inverse_scaling_factor"]) * 2)
        matrix_scaling[4, i] = math.floor(math.log2(kwargs["inverse_scaling_factor"])) * 2
    scaling_matrix = matrix_scaling.flatten()

    emit_str += [
        format_scalar_definition(
            data_in_type,
            "scaling_size",
            scaling_matrix.size,
        )
    ]
    emit_str += [
        format_vector_definition(
            data_in_type,
            "scaling_matrix",
            scaling_matrix,
        )
    ]


    # Input Matrix for Softmax
    matrix_data = np.random.randint(
        low=-4 * kwargs["inverse_scaling_factor"],
        high=4 * kwargs["inverse_scaling_factor"],
        size=(M, channel_count),
        dtype=np.int32,
    )
    input_matrix = matrix_data
    input_matrix = input_matrix.flatten()

    # Emit input matrix
    emit_str += [
        format_scalar_definition(
            "uint32_t",
            "matrix_size",
            matrix_data.size,
        )
    ]
    emit_str += [
        format_vector_definition(
            data_in_type,
            "input_matrix",
            input_matrix,
        )
    ]

    # Emit output matrix
    output_matrix = np.empty_like(matrix_data, dtype=np.int32)
    for i in range(channel_count):
        output_matrix[:, i] = integer_softmax(
            matrix_data[:, i], kwargs["inverse_scaling_factor"]
        )

    output_matrix = output_matrix.flatten()

    emit_str += [
        format_scalar_definition(
            "uint32_t",
            "output_matrix_size",
            output_matrix.size,
        )
    ]

    emit_str += [
        format_vector_definition(
            data_out_type,
            "golden_output_matrix",
            output_matrix,
        )
    ]

    # Emit the configuration for the scaling factors
    scaling_spatial_stride_src = None
    scaling_spatial_stride_dst = None
    scaling_temporal_strides_src = []
    scaling_temporal_strides_dst = []
    scaling_temporal_bounds_src = []
    scaling_temporal_bounds_dst = []

    # Input Side (Reader)
    scaling_spatial_stride_src = 8
    scaling_temporal_bounds_src = [5, 1, 1, 1, 1]
    scaling_temporal_strides_src = [4 * channel_count, 0, 0, 0, 0]

    # Output Side (Writer)
    scaling_spatial_stride_dst = 8
    scaling_temporal_bounds_dst = [0]
    scaling_temporal_strides_dst = [0]

    emit_str += [
        format_scalar_definition(
            "uint32_t",
            "spatial_stride_src_scaling",
            scaling_spatial_stride_src,
        )
    ]
    emit_str += [
        format_scalar_definition(
            "uint32_t",
            "spatial_stride_dst_scaling",
            scaling_spatial_stride_dst,
        )
    ]
    emit_str += [
        format_vector_definition(
            "uint32_t",
            "temporal_bounds_src_scaling",
            scaling_temporal_bounds_src,
        )
    ]
    emit_str += [
        format_vector_definition(
            "uint32_t",
            "temporal_bounds_dst_scaling",
            scaling_temporal_bounds_dst,
        )
    ]
    emit_str += [
        format_vector_definition(
            "uint32_t", "temporal_strides_src_scaling", scaling_temporal_strides_src
        )
    ]
    emit_str += [
        format_vector_definition(
            "uint32_t", "temporal_strides_dst_scaling", scaling_temporal_strides_dst
        )
    ]

    # Emit the configuration for XDMA
    spatial_stride_src = None
    spatial_stride_dst = None
    temporal_strides_src = []
    temporal_strides_dst = []
    temporal_bounds_src = []
    temporal_bounds_dst = []

    # Input Side (Reader)
    spatial_stride_src = 8
    temporal_bounds_src = [M, 3, 1, 1, 1]
    temporal_strides_src = [4 * channel_count, 0, 0, 0, 0]

    # Output Side (Writer)

    spatial_stride_dst = 8
    temporal_bounds_dst = [M]
    temporal_strides_dst = [4 * channel_count]

    emit_str += [
        format_scalar_definition(
            "uint32_t",
            "spatial_stride_src",
            spatial_stride_src,
        )
    ]
    emit_str += [
        format_scalar_definition(
            "uint32_t",
            "spatial_stride_dst",
            spatial_stride_dst,
        )
    ]
    emit_str += [
        format_vector_definition(
            "uint32_t",
            "temporal_bounds_src_data",
            temporal_bounds_src,
        )
    ]
    emit_str += [
        format_vector_definition(
            "uint32_t",
            "temporal_bounds_dst_data",
            temporal_bounds_dst,
        )
    ]
    emit_str += [
        format_vector_definition(
            "uint32_t", "temporal_strides_src_data", temporal_strides_src
        )
    ]
    emit_str += [
        format_vector_definition(
            "uint32_t", "temporal_strides_dst_data", temporal_strides_dst
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
    emit_str += [
        format_scalar_definition(
            "uint32_t",
            "softmax_cycles",
            M,
        )
    ]

    return emit_str


def main():

    # debugpy.listen(("0.0.0.0", 5678))
    # print("🔍 Waiting for debugger attach...")
    # debugpy.wait_for_client()  # Blocks until VS Code attaches
    # print("Debugger attached! Running program...")

    # Parsing cmd args
    parser = argparse.ArgumentParser(description="Generating data for kernels")
    parser.add_argument(
        "-c",
        "--cfg",
        type=pathlib.Path,
        required=True,
        help="Select param config file kernel",
    )
    args = parser.parse_args()

    # Load param config file
    with args.cfg.open() as f:
        param = hjson.loads(f.read())

    # Emit header file
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
