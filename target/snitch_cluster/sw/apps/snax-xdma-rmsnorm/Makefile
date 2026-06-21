# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

APP     = snax-xdma-rmsnorm
SRCS    = src/snax-xdma-rmsnorm.c
INCDIRS = data
INCDIRS += ../../snax/xdma/include

RISCV_LDFLAGS += ../../snax/xdma/build/snax-xdma-lib.o

include ../common.mk
include ./data/Makefile
$(DEP): $(DATA_H)
