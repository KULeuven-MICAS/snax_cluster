# Copyright 2020 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Nils Wistoff <nwistoff@iis.ee.ethz.ch>

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))
ROOT        := ${MKFILE_DIR}../../..

BENDER ?= bender
VIVADO ?= vivado

BENDER_TARGETS = $(shell cat $(MKFILE_DIR)../../generated/bender_targets.tmp)
BENDER_TARGETS += -t snax_fpga

snax_fpga: define_defines_includes_no_simset.tcl
	${VIVADO} -mode batch -source snax_fpga.tcl

define_defines_includes_no_simset.tcl:
	${BENDER} script vivado $(BENDER_TARGETS) --only-defines --only-includes --no-simset > $@

clean:
	rm -rf .Xil hemaia_system *.jou *.log *.str define_defines_includes_no_simset.tcl

.PHONY: program flash clean
