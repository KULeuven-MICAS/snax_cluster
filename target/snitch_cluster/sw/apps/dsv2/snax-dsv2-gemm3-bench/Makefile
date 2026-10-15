# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# GEMMs of HeMAiA's MoE prefill on array shape 3, (8, 8, 16), with INT4 weights, in isolation
# (src/snax-dsv2-gemm3-bench.c): how the weights are laid out against a row-major A, and how the
# GEMM is launched. GEMM3_BENCH_SET picks the cases (data/datagen.py):
#   unit (default)  the unit test: a shallow GEMM (K 256) in all three layouts and every way of
#                   launching four back-to-back GEMMs
#   scan            K 512, 16 rows: every layout with B 0, 8, ..., 248 bytes past A's banks
#   scanp           K 2048, 16 rows: the paired layout with B 0, 8, ..., 248 bytes past A's banks
#   amode           K 2048: A row-major, blocked or interleaved against every B layout and offsets
#   chunk           the gate|up chunk (K 2048, six column tiles) at 8 to 32 rows, every layout
#   launch          eight back-to-back GEMMs at K 2048 and K 256, 8 and 16 rows, six launches
#
#   make -C sw/apps/dsv2/snax-dsv2-gemm3-bench GEMM3_BENCH_SET=scan

APP     = snax-dsv2-gemm3-bench

GEMM3_BENCH_SET ?= unit

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

SRCS    = src/snax-dsv2-gemm3-bench.c

# Operand blobs: disassemble the code only.
OBJDUMP_FLAGS = -d

include ../../common.mk
include ../dsv2.mk

# The data's own stamp: data/Makefile regenerates data.h when the set changes.
GEMM3_BENCH_SET_STAMP = $(BUILDDIR)/gemm3-bench-set-$(GEMM3_BENCH_SET)

$(GEMM3_BENCH_SET_STAMP): | $(BUILDDIR)
	rm -f $(BUILDDIR)/gemm3-bench-set-*
	touch $@

include ./data/Makefile

# The operand blobs link in through an assembly file of .incbin directives.
SRCS += $(DATA_S)

# data.h must exist before the dependency scan.
$(DEP): $(DATA_H)
$(ELF): $(DATA_S)
