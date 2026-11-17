# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# The defaults, each overridable on the command line; TYPE is int64, i32, i16, i8 or int32,
# SEED an integer or random
M    ?= 64
N    ?= 64
TYPE ?= int64
SEED ?= 42

MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR := $(realpath $(MK_DIR))

DATAGEN_PY = $(DATA_DIR)/datagen.py
DATA_H     = $(DATA_DIR)/data.h
DATA_ARGS  = --m $(M) --n $(N) --type $(TYPE) --seed $(SEED)
