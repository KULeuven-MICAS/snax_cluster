# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>
# Robin Geens <robin.geens@esat.kuleuven.be>

APP = snax-simbacore

INCDIRS = data \
          ../../snax/simbacore/include

# Include this binary in the final build
RISCV_LDFLAGS += ../../snax/simbacore/build/snax-simbacore-lib.o

SRCS = src/snax-simbacore.c

include ../common.mk
include ./data/Makefile

$(DEP): $(DATA_H)
