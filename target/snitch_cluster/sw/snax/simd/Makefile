# Copyright 2023 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

# Usage of absolute paths is required to externally include
# this Makefile from multiple different locations

MK_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
include $(MK_DIR)/../../toolchain.mk

# Banshee runtime is not supported
ifeq ($(SELECT_RUNTIME), rtl)
RUNTIME_DIR := rtl
else
RUNTIME_DIR := rtl-generic
endif

################
## Directories #
################

# Fixed paths in repository tree
ROOT     = $(abspath $(MK_DIR)/../../../../..)
SNRT_DIR = $(ROOT)/sw/snRuntime
## Paths relative to the runtime including this Makefile
BUILDDIR = $(abspath build)
SRC_DIR  = $(abspath src)

####################
## Build variables #
####################

INCDIRS += $(abspath include)
INCDIRS += $(SNRT_DIR)/src
INCDIRS += $(SNRT_DIR)/api
INCDIRS += $(SNRT_DIR)/src/omp
INCDIRS += $(SNRT_DIR)/api/omp
INCDIRS += $(SNRT_DIR)/vendor/riscv-opcodes
INCDIRS += $(ROOT)/target/snitch_cluster/sw/runtime/common

# math.h needed by snRuntime

INCDIRS += $(SNRT_DIR)/../math/arch/riscv64/
INCDIRS += $(SNRT_DIR)/../math/arch/generic
INCDIRS += $(SNRT_DIR)/../math/src/include
INCDIRS += $(SNRT_DIR)/../math/src/internal
INCDIRS += $(SNRT_DIR)/../math/include/bits
INCDIRS += $(SNRT_DIR)/../math/include

INCDIRS += $(ROOT)/target/snitch_cluster/sw/runtime/$(RUNTIME_DIR)/src
INCDIRS += $(ROOT)/target/snitch_cluster/sw/runtime/$(RUNTIME_DIR)/include

############
## Outputs #
############

OBJS        = $(BUILDDIR)/snax-simd-lib.o
ALL_OUTPUTS = $(OBJS)


##########
## Rules #
##########

.PHONY: all
all: $(ALL_OUTPUTS)

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)

$(BUILDDIR):
	mkdir -p $@

# -MMD -MP records the headers each object actually included, so a regenerated CSR map
# (snax-*-addr.h, streamer_csr_addr_map.h) rebuilds the object instead of leaving a stale
# one behind. Without this the rule depends on the .c alone: switching cluster cfg changed
# the CSR offsets, the .c did not change, and the old object was silently linked in --
# the "make clean doesn't clean it" trap. The apps already do this (see apps/common.mk).
$(BUILDDIR)/%.o: $(SRC_DIR)/%.c | $(BUILDDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -MMD -MP -c $< -o $@

-include $(OBJS:.o=.d)
