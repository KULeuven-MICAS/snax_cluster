# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

APP     = snax-bingo-simd-bisect
SRCS    = src/snax-bingo-simd-bisect.c
INCDIRS = data

# Reader extensions on the SIMD block (hart 1). Split-cluster only.
INCDIRS += ../../snax/simd/include
RISCV_LDFLAGS += ../../snax/simd/build/snax-simd-lib.o

include ../common.mk
include ./data/Makefile
$(DEP): $(DATA_H)
