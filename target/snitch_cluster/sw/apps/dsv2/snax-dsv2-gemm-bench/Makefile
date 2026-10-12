# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# GEMM shapes in isolation on the array (src/snax-dsv2-gemm-bench.c). GEMM_BENCH_SET picks the
# cases (data/datagen.py):
#   shapes (default)  every shape, INT4 weights in the B-layout and in the paired layout
#   int8b             the gate|up shapes with INT4 weights in either layout and with INT8 weights
#   offsets           the 32-row gate|up shape with B placed 0, 32, ..., 224 bytes past A's banks
#
#   make -C sw/apps/dsv2/snax-dsv2-gemm-bench GEMM_BENCH_SET=int8b

APP     = snax-dsv2-gemm-bench

GEMM_BENCH_SET ?= shapes

INCDIRS  = data
INCDIRS += ../include
INCDIRS += ../../include
INCDIRS += ../../../snax/versacore-to/include
INCDIRS += ../../../snax/simd/include
INCDIRS += ../../../snax/xdma/include

SNAX_LIBS  = ../../../snax/versacore-to/build/snax-versacore-to-lib.o
SNAX_LIBS += ../../../snax/simd/build/snax-simd-lib.o
SNAX_LIBS += ../../../snax/xdma/build/snax-xdma-lib.o
RISCV_LDFLAGS += $(SNAX_LIBS)

SRCS    = src/snax-dsv2-gemm-bench.c

# Operand blobs: disassemble the code only.
OBJDUMP_FLAGS = -d

include ../../common.mk
include ../dsv2.mk

# The data's own stamp: data/Makefile regenerates data.h when the set changes.
GEMM_BENCH_SET_STAMP = $(BUILDDIR)/gemm-bench-set-$(GEMM_BENCH_SET)

$(GEMM_BENCH_SET_STAMP): | $(BUILDDIR)
	rm -f $(BUILDDIR)/gemm-bench-set-*
	touch $@

include ./data/Makefile

# The operand blobs link in through an assembly file of .incbin directives.
SRCS += $(DATA_S)

# data.h must exist before the dependency scan.
$(DEP): $(DATA_H)
$(ELF): $(DATA_S)
