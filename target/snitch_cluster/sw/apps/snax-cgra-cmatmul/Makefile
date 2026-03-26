# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# JYIN

APP     = snax-cgra-cmatmul

INCDIRS = ./data
INCDIRS += ../../snax/snax-cgra/include

RISCV_LDFLAGS += ../../snax/snax-cgra/build/snax-cgra-lib.o

SRCS    = src/$(APP).c

include ../common.mk

$(DEP): $(DATA_H)
