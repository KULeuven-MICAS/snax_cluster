# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Usage of absolute paths is required to externally include this Makefile
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR := $(realpath $(MK_DIR))

DATA_CFG ?= $(DATA_DIR)/params.hjson
CODEC    := $(realpath $(DATA_DIR)/../../../../../../hw/chisel/doc/crest_decompressor/crest_codec.py)

DATA_H = $(DATA_DIR)/data.h

$(DATA_H): $(DATA_DIR)/datagen.py $(DATA_CFG) $(CODEC)
	$< -c $(DATA_CFG) > $@

.PHONY: clean-data clean

clean-data:
	rm -f $(DATA_H)

clean: clean-data
