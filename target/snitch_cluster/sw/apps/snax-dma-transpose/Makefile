# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# The DMA transposes a matrix by 2-D transfers of one element per burst (README.md).
#
#   make -C sw/apps/snax-dma-transpose CFG_OVERRIDE=cfg/snax_snitch_cluster_1.hjson

APP     = snax-dma-transpose
SRCS    = src/$(APP).c
INCDIRS = data

include ../common.mk

DATA_H = $(abspath data/data.h)

$(DATA_H): data/datagen.py
	python3 $< > $@

$(DEP): $(DATA_H)

.PHONY: clean-data

clean-data:
	rm -f $(DATA_H)

clean: clean-data
