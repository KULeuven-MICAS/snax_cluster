# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

APP     = snax-flashattn

INCDIRS  = data
INCDIRS += ../../snax/gemmx/include
INCDIRS += ../../snax/simd/include

RISCV_LDFLAGS += ../../snax/gemmx/build/snax-gemmx-lib.o
RISCV_LDFLAGS += ../../snax/simd/build/snax-simd-lib.o

SRCS    = src/snax-flashattn.c

include ../common.mk
include ./data/Makefile
