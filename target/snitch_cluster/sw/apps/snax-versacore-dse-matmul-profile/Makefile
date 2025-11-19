# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

APP = snax-versacore-dse-matmul-profile

INCDIRS = data \
          ../../snax/versacore-dse/include

# Include this binary in the final build
RISCV_LDFLAGS += ../../snax/versacore-dse/build/snax-versacore-dse-lib.o

SRCS = src/snax-versacore-dse-matmul-profile.c

include ../common.mk
include ./data/Makefile

$(DEP): $(DATA_H)
