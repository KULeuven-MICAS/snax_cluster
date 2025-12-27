# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@esat.kuleuven.be>

CFG_OVERRIDE ?= cfg/snax_simbacore_cluster.hjson
CHISEL_SSM := $(shell bender path chisel-ssm)
# Path to the snitch_cluster directory (four levels up from this data dir)
CLUSTER_DIR := $(abspath $(CURR_DIR)/../../../../)

DATAGEN_PY ?= $(DATA_DIR)/datagen.py
DATAGEN_DEPS ?= $(CURR_DIR)/../../snax-simbacore-main/data/datagen_base.py
DATAGEN_DEPS += $(CURR_DIR)/../../snax-simbacore-main/data/datagen_cli.py
DATA_H     ?= $(DATA_DIR)/data.h

# Read desired workload parameters from a local params file (input)
WORKLOAD_PARAMS := $(CURR_DIR)/params_in.hjson
# Default location for generated data config in chisel-ssm (output)
DATA_CFG ?= $(CHISEL_SSM)/generated/data/$(APP_NAME)/params.hjson

# Build generator args from all top-level key:value pairs (key=value)
GENERATOR_ARGS := $(strip $(shell grep -E '^[[:space:]]*("[^"]+"|[A-Za-z0-9_]+)[[:space:]]*:' $(WORKLOAD_PARAMS) | sed -e 's://.*$$::' -e 's/[",{}]//g' -e 's/^[[:space:]]*//;s/[[:space:]]*$$//' -e 's/[[:space:]]*:[[:space:]]*/=/' | tr '\n' ' '))
GENERATOR_ARGS += name=$(APP_NAME)


.PHONY: clean-data clean

clean-data:
	rm -f $(DATA_H) $(EXTRA_CLEAN)

clean: clean-data

# Forward rule to build the Bender lock hash from the snitch_cluster
$(CLUSTER_DIR)/generated/bender_lock.hash:
	$(MAKE) -C $(CLUSTER_DIR) $(CLUSTER_DIR)/generated/bender_lock.hash


$(DATA_H): $(DATAGEN_PY) $(DATAGEN_DEPS) $(DATA_CFG)
	@echo "Generating data.h from $(DATA_CFG) with $(DATAGEN_PY)"
	$(DATAGEN_PY) --swcfg $(DATA_CFG) > $@


$(DATA_CFG): $(CLUSTER_DIR)/generated/bender_lock.hash $(WORKLOAD_PARAMS)
	@echo "Running Scala $(GENERATOR_CLASS) with args $(GENERATOR_ARGS)"
	@cd $(CHISEL_SSM) && sbt "test:runMain $(GENERATOR_CLASS) $(GENERATOR_ARGS)"
	@touch $@



