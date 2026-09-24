# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

APP     = snax-llm-qkv-to-fa
SRCS    = src/snax-llm-qkv-to-fa.c

# All four engines: VersaCore (hart 0), the SIMD block (hart 1), the xDMA (hart 2) and the
# iDMA (hart 3). Split-cluster only.
INCDIRS  = data
INCDIRS += ../../snax/versacore-to/include
INCDIRS += ../../snax/simd/include
INCDIRS += ../../snax/xdma/include

RISCV_LDFLAGS += ../../snax/versacore-to/build/snax-versacore-to-lib.o
RISCV_LDFLAGS += ../../snax/simd/build/snax-simd-lib.o
RISCV_LDFLAGS += ../../snax/xdma/build/snax-xdma-lib.o

include ../common.mk
include ./data/Makefile
# After the include that defines DATA_H: without it editing params.hjson rebuilds nothing.
$(DEP): $(DATA_H)
