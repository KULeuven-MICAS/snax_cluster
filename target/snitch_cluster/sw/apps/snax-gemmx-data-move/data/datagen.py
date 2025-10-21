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
    block_gemm_golden_model,
    postprocessing_simd_golden_model,
    align_wide_addr,
)  # noqa E402

np.random.seed(42)


# Add stdint.h header
def emit_header_file(**kwargs):
    emit_str = "#include <stdint.h>\n\n"
    emit_str += emit_data_move_data(**kwargs)
    return emit_str


MIN = -128
MAX = 127

bankWidth = 64
input_data_width = 8
output_data_width = 32
quantized_output_data_width = 8


def emit_data_move_data(**kwargs):

    meshRow = kwargs['mu']
    tileSize = kwargs['ku']
    meshCol = kwargs['nu']

    # matmul settings
    data_str = []

    data_str += [format_scalar_definition("int32_t", "Batch", 1)]
    data_str += [format_scalar_definition("int32_t", "M", kwargs["M"])]
    data_str += [format_scalar_definition("int32_t", "K", kwargs["K"])]
    data_str += [format_scalar_definition("int32_t", "N", kwargs["N"])]
    data_str += [format_scalar_definition("int32_t", "tileSize", tileSize)]
    data_str += [format_scalar_definition("int32_t", "meshRow", meshRow)]
    data_str += [format_scalar_definition("int32_t", "meshCol", meshCol)]
    data_str += [
        format_scalar_definition("int32_t", "if_separate_l1", kwargs["if_separate_l1"])
    ]

    # seperate L1 for A, B, C, D

    base_physical_addr_delta = 64
    delta_seperate_l1_a = 0
    delta_seperate_l1_b = delta_seperate_l1_a + base_physical_addr_delta
    delta_seperate_l1_b = align_wide_addr(delta_seperate_l1_b)
    delta_seperate_l1_c = delta_seperate_l1_b + base_physical_addr_delta
    delta_seperate_l1_c = align_wide_addr(delta_seperate_l1_c)
    delta_seperate_l1_d32 = delta_seperate_l1_c
    delta_seperate_l1_d32 = align_wide_addr(delta_seperate_l1_d32)
    delta_seperate_l1_d8 = delta_seperate_l1_d32
    delta_seperate_l1_d8 = align_wide_addr(delta_seperate_l1_d8)

    data_str += [
        format_scalar_definition(
            "int32_t", "delta_seperate_l1_a", delta_seperate_l1_a
        ),
        format_scalar_definition(
            "int32_t", "delta_seperate_l1_b", delta_seperate_l1_b
        ),
        format_scalar_definition(
            "int32_t", "delta_seperate_l1_c", delta_seperate_l1_c
        ),
        format_scalar_definition(
            "int32_t", "delta_seperate_l1_d8", delta_seperate_l1_d8
        ),
        format_scalar_definition(
            "int32_t", "delta_seperate_l1_d32", delta_seperate_l1_d32
        ),
    ]

    # shared L1 for A, B, C, D
    delta_shared_l1_a = 0
    delta_shared_l1_b = (
        kwargs["K"] * kwargs["M"] * (meshRow * tileSize * input_data_width / 8)
    )
    delta_shared_l1_b = align_wide_addr(delta_shared_l1_b)
    delta_shared_l1_c = delta_shared_l1_b + kwargs["K"] * kwargs["N"] * (
        meshCol * tileSize * input_data_width / 8
    )
    delta_shared_l1_c = align_wide_addr(delta_shared_l1_c)
    delta_shared_l1_d32 = delta_shared_l1_c + kwargs["M"] * kwargs["N"] * (
        meshRow * meshCol * output_data_width / 8
    )
    delta_shared_l1_d32 = align_wide_addr(delta_shared_l1_d32)
    delta_shared_l1_d8 = delta_shared_l1_d32

    data_str += [
        format_scalar_definition("int32_t", "delta_shared_l1_a", delta_shared_l1_a),
        format_scalar_definition("int32_t", "delta_shared_l1_b", delta_shared_l1_b),
        format_scalar_definition("int32_t", "delta_shared_l1_c", delta_shared_l1_c),
        format_scalar_definition("int32_t", "delta_shared_l1_d8", delta_shared_l1_d8),
        format_scalar_definition("int32_t", "delta_shared_l1_d32", delta_shared_l1_d32),
    ]

    A = np.random.randint(
        MIN, MAX, size=(kwargs["M"], kwargs["K"], meshRow, tileSize)
    ).reshape(-1)
    data_str += [format_vector_definition("int8_t", "A", A)]

    B = np.random.randint(
        MIN, MAX, size=(kwargs["K"], kwargs["N"], tileSize, meshCol)
    ).reshape(-1)
    data_str += [format_vector_definition("int8_t", "B", B)]

    C = np.random.randint(
        MIN, MAX, size=(kwargs["M"], kwargs["N"], meshRow, meshCol)
    ).reshape(-1)
    data_str += [format_vector_definition("int32_t", "C", C)]

    Psum = np.random.randint(
        MIN, MAX, size=(kwargs["M"], kwargs["N"], meshRow, meshCol)
    ).reshape(-1)
    data_str += [format_vector_definition("int32_t", "Psum", Psum)]

    D32 = np.random.randint(
        MIN, MAX, size=(kwargs["M"], kwargs["N"], meshRow, meshCol)
    ).reshape(-1)
    data_str += [format_vector_definition("int32_t", "D32", D32)]

    data_move_loop_d8 = min(kwargs["M"] * kwargs["N"] * meshRow * meshCol / 64, 1)
    data_str += [
        format_scalar_definition("int32_t", "data_move_loop_d8", data_move_loop_d8)
    ]

    D8 = np.random.randint(
        MIN, MAX, size=(kwargs["M"], kwargs["N"], meshRow, meshCol)
    ).reshape(-1)
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
