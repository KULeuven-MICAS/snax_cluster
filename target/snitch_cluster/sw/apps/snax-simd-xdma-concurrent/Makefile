# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

APP     = snax-simd-xdma-concurrent
SRCS    = src/snax-simd-xdma-concurrent.c

# This kernel drives BOTH engines, so it links both libraries. It only builds for
# the split cluster -- on an unsplit one the two engines are the same hardware and
# there is nothing to overlap.
INCDIRS += ../../snax/simd/include
INCDIRS += ../../snax/xdma/include
RISCV_LDFLAGS += ../../snax/simd/build/snax-simd-lib.o
RISCV_LDFLAGS += ../../snax/xdma/build/snax-xdma-lib.o

include ../common.mk
