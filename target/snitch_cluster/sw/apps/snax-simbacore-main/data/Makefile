# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Robin Geens <robin.geens@esat.kuleuven.be>

# Usage of absolute paths is required to externally include this Makefile
CFG_OVERRIDE ?= cfg/snax_simbacore_cluster.hjson
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
CURR_DIR := $(realpath $(MK_DIR))
CHISEL_SSM := $(shell bender path chisel-ssm)
# Path to the snitch_cluster directory (four levels up from this data dir)
CLUSTER_DIR := $(abspath $(CURR_DIR)/../../../../)
# Generated file in chisel-ssm: contains generated workload dimensions
DATA_CFG ?= $(CHISEL_SSM)/generated/data/params.hjson

DATA_H = $(CURR_DIR)/data.h

# Triggers datagen.py to regenerate data.h in case the raw data files (from Scala DataGenerator) have changed
$(DATA_H): $(CURR_DIR)/datagen.py $(CURR_DIR)/datagen_cli.py $(CURR_DIR)/datagen_base.py $(DATA_CFG) 
	$< --swcfg $(DATA_CFG) --hwcfg $(CLUSTER_DIR)/$(CFG_OVERRIDE) > $@

# TODO fetch desired workload dimensions from some config insatead of hardcoding them here
$(DATA_CFG): $(CLUSTER_DIR)/generated/bender_lock.hash
	@cd $(CHISEL_SSM) && sbt "test:runMain simbacore.DataGenerator seqLen=64 dModel=36 dtRank=24"
	@touch $@

# Forward rule to build the Bender lock hash from the snitch_cluster Makefile when invoked here
$(CLUSTER_DIR)/generated/bender_lock.hash:
	$(MAKE) -C $(CLUSTER_DIR) $(CLUSTER_DIR)/generated/bender_lock.hash


.PHONY: clean-data clean

clean-data:
	rm -f $(DATA_H) $(DATA_SUBDIR)/*.bin $(DATA_CFG)

clean: clean-data
