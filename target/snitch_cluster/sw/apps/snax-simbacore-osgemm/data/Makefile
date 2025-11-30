# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Robin Geens <robin.geens@kuleuven.be>

# Usage of absolute paths is required to externally include this Makefile
CFG_OVERRIDE ?= cfg/snax_simbacore_cluster.hjson
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
CURR_DIR := $(realpath $(MK_DIR))
CHISEL_SSM := $(shell bender path chisel-ssm)
# Generated file in chisel-ssm: contains generated workload dimensions
DATA_CFG ?= $(CHISEL_SSM)/generated/data/params.hjson

DATA_H = $(CURR_DIR)/data.h

# Triggers datagen.py to regenerate data.h in case the raw data files (from Scala DataGenerator) have changed
$(DATA_H): $(CURR_DIR)/datagen.py $(DATA_CFG)
	$< --swcfg $(DATA_CFG) --hwcfg $(ROOT)/target/snitch_cluster/$(CFG_OVERRIDE) > $@

# Generate data config by delegating to snax-simbacore-main data Makefile
$(DATA_CFG):
	$(MAKE) -C $(CURR_DIR)/../../snax-simbacore-main/data


.PHONY: clean-data clean

clean-data:
	rm -f $(DATA_H) $(DATA_SUBDIR)/*.bin $(DATA_CFG)

clean: clean-data
