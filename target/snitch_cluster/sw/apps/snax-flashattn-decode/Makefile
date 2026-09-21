# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

APP     = snax-flashattn-decode

INCDIRS  = data
INCDIRS += ../include
INCDIRS += ../../snax/versacore-to/include
INCDIRS += ../../snax/simd/include
INCDIRS += ../../snax/xdma/include

RISCV_LDFLAGS += ../../snax/versacore-to/build/snax-versacore-to-lib.o
RISCV_LDFLAGS += ../../snax/simd/build/snax-simd-lib.o
RISCV_LDFLAGS += ../../snax/xdma/build/snax-xdma-lib.o

SRCS    = src/snax-flashattn-decode.c

include ../common.mk

# Lets a measurement pass be selected without editing the source: the cluster
# contention counters are a cluster-global resource and there are only 16, so a port
# sweep is several builds of the same kernel with -DSNAX_PERF_PASS=n.
RISCV_CFLAGS += $(EXTRA_CFLAGS)
include ./data/Makefile

# Must come AFTER the include that defines DATA_H. Without it data.h is a prerequisite of
# nothing, so editing params.hjson rebuilds nothing and the app compiles against a stale
# tile shape.
$(DEP): $(DATA_H)
