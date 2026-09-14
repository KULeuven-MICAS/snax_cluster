# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Yunhao Deng <yunhao.deng@kuleuven.be>

APP     = snax-rdext-rescale-down
SRCS    = src/snax-rdext-rescale-down.c
INCDIRS = data

# Reader extensions on the xDMA, driven from the DM core.
INCDIRS += ../../snax/xdma/include
RISCV_LDFLAGS += ../../snax/xdma/build/snax-xdma-lib.o

include ../common.mk
include ./data/Makefile
$(DEP): $(DATA_H)
