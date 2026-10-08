# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

APP = snax-versacore-few-row

INCDIRS = data \
          ../../snax/versacore-to/include

RISCV_LDFLAGS += ../../snax/versacore-to/build/snax-versacore-to-lib.o

SRCS = src/snax-versacore-few-row.c

# 1.2 MiB of operand blobs: disassemble the code only.
OBJDUMP_FLAGS = -d

include ../common.mk
include ./data/Makefile

# The operand blobs link in through an assembly file of .incbin directives.
SRCS += $(DATA_S)

# data.h must exist before the dependency scan.
$(DEP): $(DATA_H)
$(ELF): $(DATA_S)
