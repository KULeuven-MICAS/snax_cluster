# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

APP     = snax-flashattn

INCDIRS  = data
INCDIRS += ../../snax/versacore-to/include
INCDIRS += ../../snax/simd/include

RISCV_LDFLAGS += ../../snax/versacore-to/build/snax-versacore-to-lib.o
RISCV_LDFLAGS += ../../snax/simd/build/snax-simd-lib.o

SRCS    = src/snax-flashattn.c

include ../common.mk
include ./data/Makefile

# Must come AFTER the include that defines DATA_H. This app was missing the line entirely,
# so data.h was not a prerequisite of anything: editing params.hjson rebuilt nothing and the
# app compiled against a stale tile shape. Every other app has it.
$(DEP): $(DATA_H)
