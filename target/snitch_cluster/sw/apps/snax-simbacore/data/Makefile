# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Robin Geens <robin.geens@esat.kuleuven.be>

# Usage of absolute paths is required to externally include this Makefile
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR := $(realpath $(MK_DIR))
DATA_CFG ?= $(DATA_DIR)/params.hjson # Generated in chisel-ssm
CFG_OVERRIDE ?= cfg/snax_simbacore_cluster.hjson

DATA_H = $(DATA_DIR)/data.h
DATA_SUBDIR = $(DATA_DIR)/generated
CHISEL_SSM_DIR := $(realpath $(DATA_DIR)/../../../../../../../chisel-ssm)

# Triggers datagen.py to regenerate data.h in case the raw data files (from Scala DataGenerator) have changed
$(DATA_H): $(DATA_DIR)/datagen.py $(DATA_DIR)/datagen_util.py $(DATA_CFG) 
	$< --swcfg $(DATA_CFG) --hwcfg $(ROOT)/target/snitch_cluster/$(CFG_OVERRIDE) > $@

$(DATA_CFG):
	@mkdir -p $(DATA_SUBDIR)
	@cd $(CHISEL_SSM_DIR) && sbt "test:runMain snax.DataGenerator seqLen=64 dModel=36"
	@touch $@

.PHONY: clean-data clean

clean-data:
	rm -f $(DATA_H) $(DATA_SUBDIR)/*.bin $(DATA_CFG)

clean: clean-data
