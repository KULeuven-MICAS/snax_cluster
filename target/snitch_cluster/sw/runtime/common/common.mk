# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

# Usage of absolute paths is required to externally include
# this Makefile from multiple different locations
MK_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
include $(MK_DIR)/../../toolchain.mk

###############
# Directories #
###############

# Fixed paths in repository tree
ROOT     = $(abspath $(MK_DIR)/../../../../..)
SNRT_DIR = $(ROOT)/sw/snRuntime

# Paths relative to the runtime including this Makefile
BUILDDIR = $(abspath build)
SRC_DIR  = $(abspath src)

###################
# Build variables #
###################

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

###########
# Outputs #
###########

OBJS        = $(addprefix $(BUILDDIR)/,$(addsuffix .o,$(basename $(notdir $(SRCS)))))
DEPS        = $(addprefix $(BUILDDIR)/,$(addsuffix .d,$(basename $(notdir $(SRCS)))))
LIB         = $(BUILDDIR)/libsnRuntime.a
DUMP        = $(BUILDDIR)/libsnRuntime.dump
ALL_OUTPUTS = $(LIB) $(DUMP)

#########
# Rules #
#########

.PHONY: all
all: $(ALL_OUTPUTS)

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)

$(BUILDDIR):
	mkdir -p $@

# -MMD -MP here for the same reason as the .c rule below, and it matters more:
# start.S reads SNRT_LOG2_STACK_SIZE, SNRT_TCDM_SIZE, CLUSTER_ADDRWIDTH and
# CLUSTER_BASE_ADDR out of snitch_cluster_defs.h / snitch_cluster_addrmap.h to
# lay out the stacks. Without dependency tracking those are invisible to make,
# so changing the stack size or the cluster's TCDM size leaves an object that
# still carves the stacks the old way -- and nothing about the app can reveal
# it, because the stacks end up somewhere the heap now believes it owns.
$(BUILDDIR)/%.o: $(SRC_DIR)/%.S | $(BUILDDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -MMD -MP -c $< -o $@

# -MMD -MP records the headers each object actually included, so a regenerated CSR map
# (snax-*-addr.h, streamer_csr_addr_map.h) rebuilds the object instead of leaving a stale
# one behind. Without this the rule depends on the .c alone: switching cluster cfg changed
# the CSR offsets, the .c did not change, and the old object was silently linked in --
# the "make clean doesn't clean it" trap. The apps already do this (see apps/common.mk).
$(BUILDDIR)/%.o: $(SRC_DIR)/%.c | $(BUILDDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -MMD -MP -c $< -o $@

# Only ever include the .d files -MMD wrote next to their object; never ask make
# to *build* one. A %.d: %.c pattern rule matches the .d of an assembled source
# against a same-named .c, and the bogus prerequisites it records then hold the
# real object up to date indefinitely.
-include $(OBJS:.o=.d)

$(LIB): $(OBJS) | $(BUILDDIR)
	$(RISCV_AR) $(RISCV_ARFLAGS) $@ $^

$(DUMP): $(LIB) | $(BUILDDIR)
	$(RISCV_OBJDUMP) -D $< > $@

