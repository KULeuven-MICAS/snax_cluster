# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

APP = snax-gemm-pipeline

INCDIRS = data \
          ../../snax/versacore-to/include

RISCV_LDFLAGS += ../../snax/versacore-to/build/snax-versacore-to-lib.o

# PIPE_FLAGS carries the pipeline shape, e.g. PIPE_FLAGS='-DNBLOCK=4 -DNK=1'
RISCV_CFLAGS += $(PIPE_FLAGS)

SRCS = src/snax-gemm-pipeline.c

include ../common.mk
include ./data/Makefile

$(DEP): $(DATA_H)
