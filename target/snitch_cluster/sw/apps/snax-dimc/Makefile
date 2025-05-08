# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Mace <ce.ma@esat.kuleuven.be>

APP     = snax-dimc

INCDIRS = data

INCDIRS += ../../snax/dimc/include

# Include this binary in the final build
RISCV_LDFLAGS += ../../snax/dimc/build/snax-dimc-lib.o

SRCS    = src/snax-dimc.c

include ../common.mk

$(DEP): $(DATA_H)
