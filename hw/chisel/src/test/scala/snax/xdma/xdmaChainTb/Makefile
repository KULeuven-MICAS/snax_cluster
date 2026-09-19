# Copyright 2026 KU Leuven.
# SPDX-License-Identifier: SHL-0.51
#
# Standalone QuestaSim flow for the N-endpoint ChainGather testbenches.
#
# Deliberately NOT the target/snitch_cluster flow: that builds the whole cluster with cores and
# needs a binary to run. Here the DUT is `snax_xdma_cluster_xdma_wrapper` instantiated N times
# behind two AXI crossbars, and the stimulus is the CSR port.
#
#   make sim-p3          # 3 endpoints, one middle hop -- the smallest reproducer
#   make sim-p2          # the control (no middle hop)
#   make sim-p4          # two middle hops, the HeMAiA failing case
#   make sim-p16         # 16 endpoints, one width
#   make sim-sweep       # THE SWEEP: P in {2,4,8,16} x {lin,mom}, with a latency table
#   make sim-tree        # tree vs chain, and whether stage 2 needs a barrier
#   make sim-star        # 4 endpoints, 3 concurrent writes into ep0
#   make sim-exchange    # 4 endpoints, 2 concurrent pairs -- every node sends AND receives
#   make sim-multi       # star, then exchange
#   make sim-all         # p2, p3, p4 in order
#   make gui-p3          # same, with the GUI
#
# `compile.tcl` depends on the actual sources, so an edit cannot leave a stale compiled design
# in place -- which is the trap that has produced false results on the HeMAiA side twice.

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))
ROOT        := $(abspath $(MKFILE_DIR)/../../../../../../../..)

# The MICAS site rc sets MGC_QUESTA_HOME and, crucially, MGLS_LICENSE_FILE. Sourcing it here
# rather than requiring the caller to have done so keeps `make sim-p3` a one-liner; without the
# licence variable vsim exits with "Invalid license environment" before compiling anything.
QUESTA_RC   ?= /esat/micas-data/data/design/scripts/questasim_2022.4.rc
QUESTA_HOME ?= $(shell . $(QUESTA_RC) >/dev/null 2>&1 && echo $$MGC_QUESTA_HOME)
QUESTA_ENV  := . $(QUESTA_RC) >/dev/null 2>&1;
VLOG        ?= $(QUESTA_HOME)/bin/vlog
VOPT        ?= $(QUESTA_HOME)/bin/vopt
VSIM        ?= $(QUESTA_HOME)/bin/vsim
VLIB        ?= $(QUESTA_HOME)/bin/vlib
BENDER      ?= bender

BUILDDIR    := $(MKFILE_DIR)work-vsim
LIB         := $(BUILDDIR)

BENDER_TARGETS := -t rtl -t simulation -t QUESTA_SIM \
                  -t snax_xdma_cluster_xdma -t xdma_chain_tb

VLOG_FLAGS  := -64 -svinputport=compat -override_timescale 1ns/1ps \
               -suppress 2583 -suppress 13314 -suppress 2286 -suppress 8386
VOPT_FLAGS  := -64 +acc
VSIM_FLAGS  := -64 -c

TB_SRCS := $(wildcard $(MKFILE_DIR)*.sv)
DUT_SRCS := \
  $(ROOT)/target/snitch_cluster/generated/snax_xdma_cluster_xdma/snax_xdma_cluster_xdma.sv \
  $(ROOT)/target/snitch_cluster/generated/snax_xdma_cluster_xdma/snax_xdma_cluster_xdma_wrapper.sv

# ---- THE GENERATED RTL IS A BUILD PRODUCT OF THE CHISEL, AND MUST BE TREATED AS ONE ---------
# `compile.tcl` already depends on DUT_SRCS, which stops a stale *vsim* compile. It does NOT stop
# a stale *generation*: DUT_SRCS are checked-in outputs with no rule tying them to the Scala they
# come from, so editing `hw/chisel/src/main/**` and re-running `make sim-tree4` happily
# re-simulates RTL built from source that no longer exists.
#
# Without it, editing the Chisel and re-running a sim target re-simulates RTL built from source
# that no longer exists, and every conclusion drawn from the result is about the wrong design.
CHISEL_SRCS := $(shell find $(ROOT)/hw/chisel/src/main/scala -name '*.scala' 2>/dev/null)
XDMA_CFG    ?= cfg/snax_xdma_cluster.hjson
# The generator needs sbt (Chisel elaboration) AND the Python stack (mako, hjson) that templates
# the wrappers -- both come from the repo's pixi environment, not from a login shell.
PIXI_BIN    := $(ROOT)/.pixi/envs/default/bin
GEN_ENV     := PATH="$(PIXI_BIN):$$PATH"

.PHONY: rtl-gen
rtl-gen:
	$(GEN_ENV) $(MAKE) -C $(ROOT)/target/snitch_cluster rtl-snax-gen CFG_OVERRIDE=$(XDMA_CFG)

# Order matters: this makes a Chisel edit invalidate the generated RTL, which invalidates
# compile.tcl, which invalidates every sim target.
$(DUT_SRCS): $(CHISEL_SRCS)
	@echo "--- generated RTL is older than the Chisel source; regenerating ---"
	$(GEN_ENV) $(MAKE) -C $(ROOT)/target/snitch_cluster rtl-snax-gen CFG_OVERRIDE=$(XDMA_CFG)

.PHONY: all clean sim-all
all: compile

$(BUILDDIR):
	mkdir -p $(BUILDDIR)
	cd $(ROOT) && $(QUESTA_ENV) $(VLIB) $(BUILDDIR)

$(BUILDDIR)/compile.tcl: $(ROOT)/Bender.yml $(ROOT)/Bender.lock $(TB_SRCS) $(DUT_SRCS) | $(BUILDDIR)
	cd $(ROOT) && $(BENDER) script vsim $(BENDER_TARGETS) \
	  --vlog-arg="$(VLOG_FLAGS) -work $(LIB)" > $@
	echo "return 0" >> $@

.PHONY: compile
compile: $(BUILDDIR)/compile.tcl
	cd $(ROOT) && $(QUESTA_ENV) $(VSIM) -64 -c -do "source $(BUILDDIR)/compile.tcl; quit" \
	  | tee $(BUILDDIR)/compile.log
	@# vsim -c prefixes every echoed line with "# ", so an anchored "^\*\* Error" never matched
	@# and a vlog failure used to sail straight through to vopt/vsim -- which then happily ran
	@# the PREVIOUS design and produced a full, plausible, completely stale set of results.
	@! grep -qE "^(# )?\*\* Error" $(BUILDDIR)/compile.log

# $(1) = testbench suffix (p2, p3, ...)
define RUN_TB
.PHONY: sim-$(1)
sim-$(1): compile
	cd $(ROOT) && $(QUESTA_ENV) $(VOPT) $(VOPT_FLAGS) -work $(LIB) tb_xdma_chaingather_$(1) \
	  -o tb_xdma_chaingather_$(1)_opt | tee $(BUILDDIR)/vopt_$(1).log
	@# same trap one stage later: vopt failing leaves the previously optimised design in place.
	@! grep -qE "^(# )?\*\* Error" $(BUILDDIR)/vopt_$(1).log
	cd $(ROOT) && $(QUESTA_ENV) $(VSIM) $(VSIM_FLAGS) -work $(LIB) tb_xdma_chaingather_$(1)_opt \
	  -do "run -all; quit" | tee $(BUILDDIR)/sim_$(1).log
	@grep -q "RESULT: PASS" $(BUILDDIR)/sim_$(1).log || \
	  (echo ""; echo "  *** $(1) FAILED -- see $(BUILDDIR)/sim_$(1).log"; exit 1)

.PHONY: gui-$(1)
gui-$(1): compile
	cd $(ROOT) && $(QUESTA_ENV) $(VOPT) $(VOPT_FLAGS) -work $(LIB) tb_xdma_chaingather_$(1) \
	  -o tb_xdma_chaingather_$(1)_opt
	cd $(ROOT) && $(QUESTA_ENV) $(VSIM) -64 -work $(LIB) tb_xdma_chaingather_$(1)_opt
endef

# ---- WAVEFORM CAPTURE -------------------------------------------------------------------
# Scoped on purpose: `log -r /*` on this design produces a WLF nobody can open. WAVE_EP picks
# the endpoint, WAVE_SCOPE can be overridden outright.
#     make wave-tree4 WAVE_EP=0
#     make vcd-tree4  WAVE_EP=0      # ... and convert to VCD for offline parsing
WAVE_EP    ?= 0
WAVE_SCOPE ?= /tb_xdma_chaingather_$(1)/i_body/gen_ep[$(WAVE_EP)]/i_ep

define WAVE_TB
.PHONY: wave-$(1)
wave-$(1): compile
	cd $(ROOT) && $(QUESTA_ENV) $(VOPT) $(VOPT_FLAGS) -work $(LIB) tb_xdma_chaingather_$(1) \
	  -o tb_xdma_chaingather_$(1)_opt | tee $(BUILDDIR)/vopt_$(1).log
	@! grep -qE "^(# )?\*\* Error" $(BUILDDIR)/vopt_$(1).log
	cd $(ROOT) && $(QUESTA_ENV) $(VSIM) $(VSIM_FLAGS) -work $(LIB) tb_xdma_chaingather_$(1)_opt \
	  -wlf $(BUILDDIR)/$(1).wlf \
	  -do "log -r $(WAVE_SCOPE)/*; run -all; quit" | tee $(BUILDDIR)/wave_$(1).log

.PHONY: vcd-$(1)
vcd-$(1): wave-$(1)
	cd $(ROOT) && $(QUESTA_ENV) $(QUESTA_HOME)/bin/wlf2vcd -o $(BUILDDIR)/$(1).vcd \
	  $(BUILDDIR)/$(1).wlf
	@echo "VCD: $(BUILDDIR)/$(1).vcd"
endef

$(foreach tb,p2 p3 p4 p16 sweep tree tree4 role bench star exchange probe_exchange probe_tree4 probe_star hang_tree4 p2mom,$(eval $(call WAVE_TB,$(tb))))

$(foreach tb,p2 p3 p4 p16 sweep tree tree4 role bench star exchange probe_exchange probe_tree4 probe_star hang_tree4 p2mom,$(eval $(call RUN_TB,$(tb))))

sim-all: sim-p2 sim-p3 sim-p4

.PHONY: sim-multi
sim-multi: sim-star sim-exchange

clean:
	rm -rf $(BUILDDIR)
