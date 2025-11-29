# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Robin Geens <robin.geens@esat.kuleuven.be>

# Usage of absolute paths is required to externally include this Makefile
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
CURR_DIR := $(realpath $(MK_DIR))
CHISEL_SSM_DIR := $(realpath $(MK_DIR)/../../../../../../chisel-ssm)
# Currently, we let the data generator depend only on the DataGenerator.scala file
CHISEL_SSM_SOURCES := $(CHISEL_SSM_DIR)/src/test/scala/simbacore/DataGenerator.scala
DATA_DIR := $(realpath $(CHISEL_SSM_DIR)/generated/data)
DATA_CFG ?= $(DATA_DIR)/params.hjson # Generated file in chisel-ssm: contains generated workload dimensions
CFG_OVERRIDE ?= cfg/snax_simbacore_cluster.hjson

DATA_H = $(CURR_DIR)/data.h

# Triggers datagen.py to regenerate data.h in case the raw data files (from Scala DataGenerator) have changed
$(DATA_H): $(CURR_DIR)/datagen.py $(CURR_DIR)/datagen_util.py $(DATA_CFG) 
	$< --swcfg $(DATA_CFG) --hwcfg $(ROOT)/target/snitch_cluster/$(CFG_OVERRIDE) > $@

# NOTE the workload dimensions are hardcoded here
$(DATA_CFG): $(CHISEL_SSM_SOURCES)
	@cd $(CHISEL_SSM_DIR) && sbt "test:runMain simbacore.DataGenerator seqLen=64 dModel=36 dtRank=24"
	@touch $@

$(CHISEL_SSM_SOURCES):
	@if [ ! -f "$@" ]; then \
		echo "Initializing/updating chisel-ssm submodule..."; \
		git -C $(CHISEL_SSM_DIR)/.. submodule update --init --recursive -- $(notdir $(CHISEL_SSM_DIR)); \
	fi

.PHONY: clean-data clean

clean-data:
	rm -f $(DATA_H) $(DATA_SUBDIR)/*.bin $(DATA_CFG)

clean: clean-data
