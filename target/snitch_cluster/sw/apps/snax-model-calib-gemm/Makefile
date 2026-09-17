# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

APP = snax-model-calib-gemm

INCDIRS = data \
          ../../snax/versacore-to/include \
          ../../snax/simd/include

RISCV_LDFLAGS += ../../snax/versacore-to/build/snax-versacore-to-lib.o
RISCV_LDFLAGS += ../../snax/simd/build/snax-simd-lib.o

SRCS = src/snax-model-calib-gemm.c

include ../common.mk
include ./data/Makefile

$(DEP): $(DATA_H)
