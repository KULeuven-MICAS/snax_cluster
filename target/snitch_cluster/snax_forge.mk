# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Running a snax-forge bundle's program on this cluster's RTL, with the existing targets only
# (no podman: rtl-gen, and for vsim vsim_preparation, run first where snax-forge's docs/RUN.md
# says). Included by the Makefile.
#
#   CFG_OVERRIDE  the bundle's cluster file (a fixture under cfg/snax_forge/ or an absolute path)
#   FORGE         the program's folder (main.c, data.h, include/), by default CFG_OVERRIDE's
#   ELF           an ELF to run; unset, snax-forge-run runs the app built from FORGE
#   SIM           vlt (default) or vsim
#
#   snax-forge-sim  bin/snitch_cluster.$(SIM); for vsim, vsim_preparation must have run
#   snax-forge-sw   sw/apps/snax-forge for FORGE, once the #define lines of FORGE's
#                   streamer_csr_addr_map.h equal those of the one snaxgen wrote
#   snax-forge-run  the ELF on bin/snitch_cluster.$(SIM), its output in logs/snax-forge-$(SIM).txt
#                   beside the run's traces; fails on the program's nonzero exit code

SIM   ?= vlt
FORGE ?= $(dir $(abspath $(CFG_OVERRIDE)))
override FORGE := $(abspath $(FORGE))

SNAX_FORGE_APP    = $(SW_DIR)/apps/snax-forge
SNAX_FORGE_HEADER = $(SW_DIR)/snax/snax_forge/include/streamer_csr_addr_map.h
SNAX_FORGE_LOG    = $(LOGS_DIR)/snax-forge-$(SIM).txt
include $(MKFILE_DIR)$(SNAX_FORGE_APP)/build-dir.mk
SNAX_FORGE_RUN_ELF = $(if $(ELF),$(abspath $(ELF)),$(SNAX_FORGE_ELF))

ifneq ($(filter snax-forge-%,$(MAKECMDGOALS)),)
ifneq ($(words $(filter vlt vsim,$(SIM))) $(words $(SIM)),1 1)
$(error SIM=$(SIM): use vlt or vsim)
endif
ifeq ($(strip $(CFG_OVERRIDE)),)
$(error CFG_OVERRIDE is not set: give the bundle's cluster file)
endif
endif

.PHONY: snax-forge-sim snax-forge-sw snax-forge-run

snax-forge-sim:
ifeq ($(SIM),vsim)
	@test -f $(VSIM_BUILDDIR)/compile.vsim.tcl -a -f work/lib/libfesvr.a || \
		{ echo "snax-forge-sim: run vsim_preparation first (snax-forge docs/RUN.md)"; exit 1; }
endif
	$(MAKE) bin/snitch_cluster.$(SIM)

# The bundle's header is the one the ELF uses; snaxgen's is the RTL's (comments differ)
snax-forge-sw: $(TARGET_C_HDRS)
	@test -f $(SNAX_FORGE_HEADER) || \
		{ echo "snax-forge-sw: no $(SNAX_FORGE_HEADER): run rtl-gen or sw-snax-gen first"; exit 1; }
	mkdir -p $(SNAX_FORGE_BUILD)
	grep '^[[:space:]]*#define' $(FORGE)/include/streamer_csr_addr_map.h > $(SNAX_FORGE_BUILD)/bundle.defines
	grep '^[[:space:]]*#define' $(SNAX_FORGE_HEADER) > $(SNAX_FORGE_BUILD)/snaxgen.defines
	@diff $(SNAX_FORGE_BUILD)/bundle.defines $(SNAX_FORGE_BUILD)/snaxgen.defines || \
		{ echo "snax-forge-sw: $(FORGE)/include/streamer_csr_addr_map.h and $(SNAX_FORGE_HEADER) differ"; exit 1; }
	$(MAKE) -C $(SW_DIR) math runtime/$(or $(strip $(SELECT_RUNTIME)),rtl-generic)
	$(MAKE) -C $(SNAX_FORGE_APP) FORGE=$(FORGE)

# Questa exits 0 whatever the program returns, so its exit code is read from the tb's last line
snax-forge-run: $(if $(ELF),,snax-forge-sw)
	@test -x bin/snitch_cluster.$(SIM) || \
		{ echo "snax-forge-run: no bin/snitch_cluster.$(SIM): make snax-forge-sim first"; exit 1; }
	@test -f $(SNAX_FORGE_RUN_ELF) || { echo "snax-forge-run: no ELF $(SNAX_FORGE_RUN_ELF)"; exit 1; }
	mkdir -p $(LOGS_DIR)
	bash -c 'set -o pipefail; bin/snitch_cluster.$(SIM) $(SNAX_FORGE_RUN_ELF) $(if $(filter vsim,$(SIM)),"" -batch) 2>&1 | tee $(SNAX_FORGE_LOG)'
ifeq ($(SIM),vsim)
	@grep -q '\[SUCCESS\] Program finished successfully' $(SNAX_FORGE_LOG) || \
		{ echo "snax-forge-run: the program did not finish successfully ($(SNAX_FORGE_LOG))"; exit 1; }
endif
