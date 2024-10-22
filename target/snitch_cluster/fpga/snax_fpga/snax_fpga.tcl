# Copyright 2020 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Nils Wistoff <nwistoff@iis.ee.ethz.ch>

set nproc [exec nproc]

# Create project
set project snax_fpga

create_project $project ./$project -force -part xcvp1802-lsvc4072-2MP-e-S
set_property board_part xilinx.com:vpk180:part0:1.2 [current_project]
set_property XPM_LIBRARIES XPM_MEMORY [current_project]

set_property ip_repo_paths ../snax_ip [current_project]
update_ip_catalog

# Create block design
source snax_fpga_bd.tcl

# Add constraint files
add_files -fileset constrs_1 -norecurse snax_impl.xdc
import_files -fileset constrs_1 snax_impl.xdc
set_property used_in_synthesis false [get_files snax_fpga/snax_fpga.srcs/constrs_1/imports/snax_fpga/snax_impl.xdc]

# Generate wrapper
make_wrapper -files [get_files ./snax_fpga/snax_fpga.srcs/sources_1/bd/snax_top/snax_top.bd] -top
add_files -norecurse ./snax_fpga/snax_fpga.gen/sources_1/bd/snax_top/hdl/snax_top_wrapper.v
update_compile_order -fileset sources_1

# Create runs
generate_target all [get_files ./snax_fpga/snax_fpga.srcs/sources_1/bd/snax_top/snax_top.bd]
export_ip_user_files -of_objects [get_files ./snax_fpga/snax_fpga.srcs/sources_1/bd/snax_top/snax_top.bd] -no_script -sync -force -quiet
create_ip_run [get_files -of_objects [get_fileset sources_1] ./snax_fpga/snax_fpga.srcs/sources_1/bd/snax_top/snax_top.bd]

# Re-add hemaia chip includes
set build snax_fpga

export_ip_user_files -of_objects [get_ips snax_xilinx_0] -no_script -sync -force -quiet
eval [exec sed {s/current_fileset/get_filesets snax_top_snax_xilinx_0_0/} define_defines_includes_no_simset.tcl]

# Do NOT insert BUFGs on high-fanout nets (e.g. reset). This will backfire during placement.
# set_param logicopt.enableBUFGinsertHFN no

# OOC synthesis of changed IP
set synth_runs [get_runs *synth*]
set synth_1_idx [lsearch $synth_runs "synth_1"]
set all_ooc_synth [lreplace $synth_runs $synth_1_idx $synth_1_idx]
set runs_queued {}
foreach run $all_ooc_synth {
    if {[get_property PROGRESS [get_run $run]] != "100%"} {
        puts "Launching run $run"
        lappend runs_queued $run
        # Default synthesis strategy
        set_property strategy Flow_PerfOptimized_high [get_runs $run]
    } else {
        puts "Skipping 100% complete run: $run"
    }
}
if {[llength $runs_queued] != 0} {
    reset_run $runs_queued
    launch_runs $runs_queued -jobs ${nproc}
    puts "Waiting on $runs_queued"
    foreach run $runs_queued {
        wait_on_run $run
    }
    # reset main synthesis
    reset_run synth_1
}

# top-level synthesis
set run synth_1
set_property strategy Flow_PerfOptimized_high [get_runs $run]
if {[get_property PROGRESS [get_run $run]] != "100%"} {
    puts "Launching run $run"
    reset_run $run
    set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs $run]
    launch_runs $run -jobs ${nproc}
    wait_on_run $run
} else {
    puts "Skipping 100% complete run: $run"
}

# Implement
set_property strategy Peformance_ExplorePostRoutePhysOpt [get_runs impl_1]
launch_runs impl_1 -jobs ${nproc}
wait_on_run impl_1

# Generate Bitstream
launch_runs impl_1 -to_step write_bitstream -jobs ${nproc}
wait_on_run impl_1

# Reports
proc write_report_timing { build project run name } {
    exec mkdir -p ${build}/${project}.reports

    # Global timing report
    report_timing_summary -nworst 20 -file ${build}/${project}.reports/${name}_timing_${run}.rpt

    # timing specific to hemaia
    catch {
        report_timing_summary -nworst 20 -cells [get_cells -hierarchical -filter { ORIG_REF_NAME =~ occamy*_top }] \
            -file ${build}/${project}.reports/${name}_timing_${run}_chip.rpt
    }
    # 20 worst setup times
    catch {
        report_timing_summary -nworst 20 -setup -cells [get_cells -hierarchical -filter { ORIG_REF_NAME =~ occamy*_top }] \
            -file ${build}/${project}.reports/${name}_timing_${run}_chip_setup.rpt
    }
}

proc write_report_util { build project run name } {
    exec mkdir -p ${build}/${project}.reports
    report_utilization -file ${build}/${project}.reports/${name}_util_${run}.rpt
    report_utilization -hierarchical -hierarchical_percentages -file ${build}/${project}.reports/${name}_utilhierp_${run}.rpt
    report_utilization -hierarchical -file ${build}/${project}.reports/${name}_utilhier_${run}.rpt
    report_utilization -hierarchical -hierarchical_percentages -hierarchical_depth 5 -file ${build}/${project}.reports/${name}_utilhierpf_${run}.rpt
}

if {[get_property PROGRESS [get_run impl_1]] == "100%"} {
    # implementation report
    open_run impl_1
    #write_report_timing ${build} ${project} impl_1 2_post_impl
    #write_report_util ${build} ${project} impl_1 2_post_impl
    close_design
} else {
    puts "ERROR: Something went wrong in implementation, it should have 100% PROGRESS by now."
    exit 2
}
