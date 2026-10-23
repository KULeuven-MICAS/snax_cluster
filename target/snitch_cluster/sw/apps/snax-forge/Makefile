# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# The program of a snax-forge bundle or fixture: $(FORGE)/main.c, with $(FORGE) (its data.h) and
# $(FORGE)/include (streamer_csr_addr_map.h, <instance>_csr.h) first on the include path. No
# datagen: data.h is the bundle's. FORGE defaults to the folder of CFG_OVERRIDE (relative to
# target/snitch_cluster, or absolute); each FORGE has its own build folder (build-dir.mk), and the
# ELF depends on every file in FORGE. Modelled on snax-gvsoc's sw/apps/forge/Makefile;
# target/snitch_cluster/snax_forge.mk runs it.

APP = snax-forge

SNITCH_CLUSTER_DIR := $(abspath $(CURDIR)/../../..)
FORGE ?= $(dir $(abspath $(if $(filter /%,$(CFG_OVERRIDE)),,$(SNITCH_CLUSTER_DIR)/)$(CFG_OVERRIDE)))
override FORGE := $(abspath $(FORGE))
ifneq ($(MAKECMDGOALS),clean)
ifeq ($(wildcard $(FORGE)/main.c),)
$(error no main.c in $(FORGE): FORGE (or CFG_OVERRIDE's folder) is a snax-forge bundle's program folder)
endif
endif

include build-dir.mk
override BUILDDIR = $(SNAX_FORGE_BUILD)

INCDIRS = $(FORGE)
INCDIRS += $(FORGE)/include
SRCS    = $(FORGE)/main.c

include ../common.mk

$(ELF): $(wildcard $(FORGE)/* $(FORGE)/include/*)
