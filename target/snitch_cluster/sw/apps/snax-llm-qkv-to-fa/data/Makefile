# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR := $(realpath $(MK_DIR))

DATA_CFG ?= $(DATA_DIR)/params.hjson
DATA_H = $(DATA_DIR)/data.h

# The mesh comes from the HARDWARE cfg, so a cfg switch must regenerate data.h.
HW_CFG := $(ROOT)/target/snitch_cluster/$(CFG_OVERRIDE)

$(DATA_H): $(DATA_DIR)/datagen.py $(DATA_CFG) $(HW_CFG)
	$< --swcfg $(DATA_CFG) --hwcfg $(HW_CFG) > $@

.PHONY: clean-data clean
clean-data:
	rm -f $(DATA_H)
clean: clean-data
