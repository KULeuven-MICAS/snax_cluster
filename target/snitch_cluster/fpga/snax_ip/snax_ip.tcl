# Copyright 2024 KU Leuven and ETH Zurich.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Nils Wistoff <nwistoff@iis.ee.ethz.ch>
# Yunhao Deng <yunhao.deng@kuleuven.be>

# Create project
set project snax

create_project $project ./snax -force -part xcvp1802-lsvc4072-2MP-e-S
set_property XPM_LIBRARIES XPM_MEMORY [current_project]

# Define sources
source define-sources.tcl

# Add constraints
set constraint_file synth_constraints.xdc
if {[file exists ${constraint_file}]} {
    add_files -fileset constrs_1 -norecurse ${constraint_file}
    set_property USED_IN {synthesis out_of_context} [get_files ${constraint_file}]
}

# Buggy Vivado doesn't like these files. That's ok, we don't need them anyways.
set_property IS_ENABLED 0 [get_files -regex .*/axi_intf.sv]
set_property IS_ENABLED 0 [get_files -regex .*/reg_intf.sv]

# Package IP
set_property top snax_xilinx [current_fileset]

update_compile_order -fileset sources_1
# This is just a quick synthesize to ensure there is no errors in source code
synth_design -rtl -name rtl_1

ipx::package_project -root_dir . -vendor MICAS_KUL -library user -taxonomy /UserIP -set_current true

# Clock interface
ipx::infer_bus_interface clk_i xilinx.com:signal:clock_rtl:1.0 [ipx::current_core]

# Reset interface
ipx::infer_bus_interface rst_ni xilinx.com:signal:reset_rtl:1.0 [ipx::current_core]

# Associate clock to AXI interfaces
ipx::associate_bus_interfaces -busif s_axi_xilinx -clock clk_i [ipx::current_core]


# Export
set_property core_revision 1 [ipx::current_core]
ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::save_core [ipx::current_core]
ipx::check_integrity [ipx::current_core]
ipx::save_core [ipx::current_core]
