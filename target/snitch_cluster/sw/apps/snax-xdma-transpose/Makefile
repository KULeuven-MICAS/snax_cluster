# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Yunhao Deng <yunhao.deng@kuleuven.be>

APP     = snax-xdma-transpose
SRCS    = src/snax-xdma-transpose.c
INCDIRS = data

# The Transposer is a pure wire permutation -- matrixOutput(j)(i) := matrixInput(i)(j)
# and nothing else. That is data movement, not compute, so it stays on the xDMA
# on every cluster shape: no compat shim, no SIMD library.
INCDIRS += ../../snax/xdma/include
RISCV_LDFLAGS += ../../snax/xdma/build/snax-xdma-lib.o

include ../common.mk
include ./data/Makefile
$(DEP): $(DATA_H)
