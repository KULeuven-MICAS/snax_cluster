# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

APP     = snax-model-calib
SRCS    = src/snax-model-calib.c

INCDIRS += ../../snax/simd/include
INCDIRS += ../../snax/xdma/include
RISCV_LDFLAGS += ../../snax/simd/build/snax-simd-lib.o
RISCV_LDFLAGS += ../../snax/xdma/build/snax-xdma-lib.o

include ../common.mk
