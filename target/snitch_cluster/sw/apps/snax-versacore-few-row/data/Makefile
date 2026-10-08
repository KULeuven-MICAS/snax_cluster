# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR := $(realpath $(MK_DIR))

DATA_CFG ?= $(DATA_DIR)/params.hjson
DATA_H    = $(DATA_DIR)/data.h
# The operands are linked as raw blobs (.incbin), not C text.
BLOB_DIR  = $(abspath build)/data
DATA_S    = $(BLOB_DIR)/blobs.S

# The generator reads the array shapes from the hardware cfg, and the layouts and the blob
# emitter from the dsv2 apps' util package.
HW_CFG    := $(ROOT)/target/snitch_cluster/$(CFG_OVERRIDE)
DSV2_UTIL := $(DATA_DIR)/../../dsv2/util
UTIL_PY   := $(DSV2_UTIL)/layout.py $(DSV2_UTIL)/emit.py

$(DATA_H): $(DATA_DIR)/datagen.py $(DATA_CFG) $(HW_CFG) $(UTIL_PY)
	mkdir -p $(BLOB_DIR)
	$< --swcfg $(DATA_CFG) --hwcfg $(HW_CFG) --header $@ --blob-dir $(BLOB_DIR)

$(DATA_S): $(DATA_H)

.PHONY: clean-data clean

clean-data:
	rm -f $(DATA_H)
	rm -rf $(BLOB_DIR)

clean: clean-data
