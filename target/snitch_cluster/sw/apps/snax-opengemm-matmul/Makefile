# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX - License - Identifier : Apache - 2.0
#
# Xiaoling Yi < xiaoling.yi @esat.kuleuven.be>

APP = snax-opengemm-matmul

INCDIRS = data \
          ../../snax/opengemm/include

# Include this binary in the final build
RISCV_LDFLAGS += ../../snax/opengemm/build/snax-opengemm-lib.o

SRCS = src/snax-opengemm-matmul.c

include ../common.mk
include ./data/Makefile

$(DEP): $(DATA_H)
