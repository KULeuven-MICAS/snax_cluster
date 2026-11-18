# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# The defaults, each overridable on the command line; TYPE is int8 or int64 (the copy moves
# in_t, so in_t and acc_t must be one type), SEED an integer or random
M    ?= 64
N    ?= 64
TYPE ?= int8
SEED ?= 42

MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR := $(realpath $(MK_DIR))

DATAGEN_PY = $(DATA_DIR)/datagen.py
DATA_H     = $(DATA_DIR)/data.h
DATA_ARGS  = --m $(M) --n $(N) --type $(TYPE) --seed $(SEED)
