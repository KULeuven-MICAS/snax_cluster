# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Mace <ce.ma@esat.kuleuven.be>

APP     = snax-dimc-conv

# Define the GitHub repository URL and target directory
GIT_REPO = https://github.com/KULeuven-MICAS/hemaia_dimc_data.git
DATA_DIR = data
CLONE_DIR = hemaia_dimc_data  # The original repo name after cloning

# Ensure data directory is cloned before anything else
$(DATA_DIR):
	@echo "Checking if data directory exists..."
	@if [ ! -d "$(DATA_DIR)/.git" ]; then \
		echo "Cloning repository..."; \
		git clone $(GIT_REPO); \
		echo "Renaming repository to $(DATA_DIR)..."; \
		mv $(CLONE_DIR) $(DATA_DIR); \
	else \
		echo "Repository already cloned."; \
	fi

# Ensure data.h is available before proceeding
$(DATA_DIR)/data.h: $(DATA_DIR)
	@if [ ! -f "$(DATA_DIR)/data.h" ]; then \
		echo "Error: data.h not found in $(DATA_DIR)!"; \
		exit 1; \
	fi

# Add INCDIRS only after cloning is complete
INCDIRS += $(DATA_DIR)
INCDIRS += ../../snax/dimc/include

# Include this binary in the final build
RISCV_LDFLAGS += ../../snax/dimc/build/snax-dimc-lib.o

SRCS    = src/snax-dimc-conv.c

include ../common.mk

$(DEP): $(DATA_DIR)/data.h

# Ensure cloning happens first before compiling
all: $(DATA_DIR)/data.h