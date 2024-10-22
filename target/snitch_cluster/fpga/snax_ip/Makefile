# Copyright 2020 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Nils Wistoff <nwistoff@iis.ee.ethz.ch>

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))
ROOT        := ${MKFILE_DIR}../../..
DEBUG       ?= 1
EXT_JTAG    ?= 1

VIVADO ?= vivado
BENDER ?= bender

BENDER_TARGETS = $(shell cat $(MKFILE_DIR)../../generated/bender_targets.tmp)
BENDER_TARGETS += -t snax_fpga

snax_ip: define-sources.tcl
	${VIVADO} -mode batch -source snax_ip.tcl

define-sources.tcl:
	${BENDER} script vivado $(BENDER_TARGETS) > $@

clean:
	rm -rf .Xil snax vivado* define-sources.tcl component.xml xgui *.jou *.log
