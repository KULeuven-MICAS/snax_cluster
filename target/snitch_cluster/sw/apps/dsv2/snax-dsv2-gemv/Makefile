# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

APP     = snax-dsv2-gemv

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

SRCS    = src/snax-dsv2-gemv.c

# 36 MiB of weight blobs: disassemble the code only.
OBJDUMP_FLAGS = -d

include ../../common.mk
include ../dsv2.mk
include ./data/Makefile

# The weight blobs link in through an assembly file of .incbin directives.
SRCS += $(DATA_S)

# data.h must exist before the dependency scan, and an ELF must relink when a library
# object it links changes -- common.mk lists only the runtime archives.
$(DEP): $(DATA_H)
$(ELF): $(DATA_S) $(SNAX_LIBS)
