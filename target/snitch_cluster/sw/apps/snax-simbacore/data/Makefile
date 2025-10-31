# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

# Usage of absolute paths is required to externally include this Makefile
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR := $(realpath $(MK_DIR))

DATA_CFG ?= $(DATA_DIR)/params.hjson
CFG_OVERRIDE ?= cfg/snax_simbacore_cluster.hjson

DATA_H = $(DATA_DIR)/data.h
DATA_SUBDIR = $(DATA_DIR)/generated

# NOTE add more datafile here, to be generated externally and parsed by datagen.py
DATA_SUBFOLDER_FILES = $(DATA_SUBDIR)/A.bin $(DATA_SUBDIR)/B.bin $(DATA_SUBDIR)/C.bin $(DATA_SUBDIR)/D.bin

$(DATA_H): $(DATA_DIR)/datagen.py $(DATA_CFG) $(DATA_SUBFOLDER_FILES)
	$< --swcfg $(DATA_CFG) --hwcfg $(ROOT)/target/snitch_cluster/$(CFG_OVERRIDE) > $@

$(DATA_SUBDIR)/%.bin: $(DATA_CFG)
	@mkdir -p $(DATA_SUBDIR)
	CUR_DIR := $(shell pwd)
	CHISEL_SSM_DIR := $(realpath $(DATA_DIR)/../../../../../../../chisel-ssm)
	cd $(CHISEL_SSM_DIR)
	sbt "test:runMain snax.MatmulDataGenerator $(DATA_CFG)"
	cd $(CUR_DIR)

.PHONY: clean-data clean

clean-data:
	rm -f $(DATA_H) $(DATA_SUBFOLDER_FILES)

clean: clean-data
