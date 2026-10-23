# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# The ELF of the program folder FORGE (an absolute path): build/<its name>-<cksum of its path>/,
# so two folders (a fixture's, a bundle's of the same design) never share an ELF path. Read by
# this folder's Makefile and by target/snitch_cluster/snax_forge.mk.

SNAX_FORGE_APP_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SNAX_FORGE_BUILD    = $(SNAX_FORGE_APP_DIR)build/$(notdir $(FORGE))-$(firstword $(shell printf '%s' '$(FORGE)' | cksum))
SNAX_FORGE_ELF      = $(SNAX_FORGE_BUILD)/snax-forge.elf
