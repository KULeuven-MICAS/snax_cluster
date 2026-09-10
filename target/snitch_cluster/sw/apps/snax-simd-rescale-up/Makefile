# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Yunhao Deng <yunhao.deng@kuleuven.be>

APP     = snax-simd-rescale-up
SRCS    = src/snax-simd-rescale-up.c
INCDIRS = data

# The reader extensions this kernel drives live on the SIMD block for the split
# cluster and on the xDMA everywhere else; snax-simd-compat.h hides the difference.
ifeq ($(CFG_OVERRIDE), cfg/snax_xdma_split_cluster.hjson)
INCDIRS += ../../snax/simd/include
RISCV_LDFLAGS += ../../snax/simd/build/snax-simd-lib.o
else
INCDIRS += ../../snax/xdma/include
RISCV_LDFLAGS += ../../snax/xdma/build/snax-xdma-lib.o
endif

include ../common.mk
include ./data/Makefile
$(DEP): $(DATA_H)
