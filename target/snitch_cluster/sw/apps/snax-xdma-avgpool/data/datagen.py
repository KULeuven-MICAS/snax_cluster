#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Jonas Crols <jonas.crols@student.kuleuven.be>

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402

np.random.seed(320)


# Add stdint.h header
def emit_header_file(**kwargs):
    emit_str = ["#include <stdint.h>"]
    emit_str += emit_elementwise_add_data(**kwargs)
    return "\n\n".join(emit_str)


def emit_elementwise_add_data(**kwargs):
    tile_width = None
    element_width = kwargs["BIT_WIDTH"]
    
    data_in_type = "uint8_t"
    data_out_type = "uint8_t"

    emit_str = []
    padded_M = kwargs["M"] 
    padded_N = kwargs["N"]
    channel_count = kwargs["channels"]
    m_kernel = kwargs["m_kernel"]
    n_kernel = kwargs["n_kernel"]
    m_stride = kwargs["m_stride"]
    n_stride = kwargs["n_stride"]

    # First input matrix for elementwise add
    matrix_data = np.zeros((padded_M, padded_N), dtype=np.uint8)
    matrix_data[: kwargs["M"], : kwargs["N"]] = np.random.randint(
        low=128, high=127 , size=(kwargs["M"], kwargs["N"], channel_count), dtype=np.uint8
    )
    input_matrix = matrix_data
    input_matrix = input_matrix.flatten()

    # Emit input matrix
    emit_str += [
        format_scalar_definition("uint8_t", "matrix_size", matrix_data.size)
    ]
    emit_str += [format_vector_definition(data_in_type, "input_matrix", input_matrix)]

    # Emit output matrix
    output_matrix = maxpool_golden(
        a_vals=matrix_data,
        m=padded_M,
        n=padded_N,
        channels=channel_count,
        m_kernel=m_kernel,
        n_kernel=n_kernel,
        m_stride=m_stride,
        n_stride=n_stride,
    )

    output_matrix = output_matrix.flatten()
    emit_str += [
        format_vector_definition(data_out_type, "golden_output_matrix", output_matrix)
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
    temporal_bounds_src = [
        3,3,13,13,1
    ]
    temporal_strides_src = [
        64, 1792, 128, 3584, 0
    ]

    # Output Side (Writer)

    spatial_stride_dst = 8
    temporal_bounds_dst = [
        169
    ]
    temporal_strides_dst = [
        64,
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


def maxpool_golden(
    a_vals: np.ndarray,
    m: int,
    n: int,
    channels: int,
    m_kernel: int,
    n_kernel: int,
    m_stride: int,
    n_stride: int,
) -> np.ndarray:
    """
    Compute the golden output for maxpool operation.
    This function simulates the maxpool operation on the input tensor.
    """
    output = np.empty(
        (
            ((m - m_kernel) // m_stride + 1),
            ((n - n_kernel) // n_stride + 1),
            channels,
        ),
        dtype=np.int8,
    )
    # Iterate over each channel and apply max pooling
    for i in range(0, ((m - m_kernel) // m_stride + 1) * m_stride, m_stride):
        for j in range(0, ((n - n_kernel) // n_stride + 1) * n_stride, n_stride):
            for c in range(channels):
                # Extract the kernel region
                kernel_region = a_vals[i : i + m_kernel, j : j + n_kernel, c]
                # Compute the maximum value in the kernel region
                max_value = (int(np.sum(kernel_region)) * 1242757) >> 25  # Scale down to uint8 range
                output[i // m_stride, j // n_stride, c] = max_value
    return output


def main():
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
