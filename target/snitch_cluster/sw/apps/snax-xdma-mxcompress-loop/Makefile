# Copyright 2026 KU Leuven. SPDX-License-Identifier: Apache-2.0

APP     = snax-xdma-mxcompress-loop
SRCS    = src/snax-xdma-mxcompress-loop.c
INCDIRS = data
INCDIRS += ../../snax/xdma/include

RISCV_LDFLAGS += ../../snax/xdma/build/snax-xdma-lib.o

include ../common.mk
include ./data/Makefile
$(DEP): $(DATA_H)
