# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

APP     = snax-simd-softmax
SRCS    = src/snax-simd-softmax.c
INCDIRS = data

# Reader extensions on the SIMD block (hart 1). Split-cluster only.
INCDIRS += ../../snax/simd/include
RISCV_LDFLAGS += ../../snax/simd/build/snax-simd-lib.o
# ...and the xDMA block (hart 2), which owns the Transposer: the transposed path needs
# x^T, and producing it is a data movement on the other engine, not SIMD work.
INCDIRS += ../../snax/xdma/include
RISCV_LDFLAGS += ../../snax/xdma/build/snax-xdma-lib.o

include ../common.mk
include ./data/Makefile
$(DEP): $(DATA_H)
