# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# What every snitch-forge program shares, included by <kernel>/Makefile after its data/Makefile
# (which sets DATAGEN_PY, DATA_H and DATA_ARGS from its defaults) and before ../../common.mk.
#
#   make -C <kernel> CFG_OVERRIDE=cfg/snax_snitch_cluster_<1|5|9>.hjson [N=...] [TYPE=...] [SEED=...]
#
# builds build/<kernel>.elf, which prints and checks; QUIET=1 builds build/<kernel>_quiet.elf
# with -DQUIET, which writes the section marks to CSR 0x7c5 (obs_marks.json) and prints nothing.
# Both come from src/<kernel>.c and data/data.h, so they sit side by side. data.h is written
# again when a value in DATA_ARGS changes (build/datagen.stamp holds the last ones), and on
# every build with SEED=random. Build right before each run: the runtime, and so the ELF, is the
# last CFG_OVERRIDE's.

SF_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

# common.mk's all, though FORCE below comes first
.DEFAULT_GOAL := all

ifeq ($(QUIET),1)
APP = $(KERNEL)_quiet
RISCV_CFLAGS += -DQUIET
else
APP = $(KERNEL)
endif

SRCS     = $(CURDIR)/src/$(KERNEL).c
INCDIRS += $(DATA_DIR) $(SF_DIR)

DATA_STAMP = $(CURDIR)/build/datagen.stamp

.PHONY: FORCE clean-data
FORCE:

$(DATA_STAMP): FORCE
	@mkdir -p $(dir $@)
	@echo '$(DATA_ARGS)' | cmp -s - $@ || echo '$(DATA_ARGS)' > $@

$(DATA_H): $(DATAGEN_PY) $(SF_DIR)/snitch_forge_data.py $(DATA_STAMP) $(if $(filter random,$(SEED)),FORCE)
	$(DATAGEN_PY) $(DATA_ARGS) > $@.tmp
	mv $@.tmp $@

clean-data:
	rm -f $(DATA_H) $(DATA_H).tmp

clean: clean-data
