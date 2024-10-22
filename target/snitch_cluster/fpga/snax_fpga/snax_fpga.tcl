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

set_property ip_repo_paths ../snax_ip [current_project]update_ip_catalog

# Create block design
source hemaia_system_vpk180_bd.tcl

# Add constraint files
add_files -fileset constrs_1 -norecurse hemaia_system_vpk180_impl.xdc
import_files -fileset constrs_1 hemaia_system_vpk180_impl.xdc
set_property used_in_synthesis false [get_files hemaia_system/hemaia_system.srcs/constrs_1/imports/hemaia_system/hemaia_system_vpk180_impl.xdc]
if { $EXT_JTAG } {
    add_files -fileset constrs_1 -norecurse hemaia_system_vpk180_impl_ext_jtag.xdc
    import_files -fileset constrs_1 hemaia_system_vpk180_impl_ext_jtag.xdc
    set_property used_in_synthesis false [get_files hemaia_system/hemaia_system.srcs/constrs_1/imports/hemaia_system/hemaia_system_vpk180_impl.xdc]
} else {
    delete_bd_objs [get_bd_nets -of_objects [get_bd_ports "jtag_tck_i jtag_tdi_i jtag_tdo_o jtag_tms_i" ]]
    delete_bd_objs [get_bd_ports jtag_*]
}

# Generate wrapper
make_wrapper -files [get_files ./hemaia_system/hemaia_system.srcs/sources_1/bd/hemaia_system/hemaia_system.bd] -top
add_files -norecurse ./hemaia_system/hemaia_system.gen/sources_1/bd/hemaia_system/hdl/hemaia_system_wrapper.v
update_compile_order -fileset sources_1

# Create runs
generate_target all [get_files ./hemaia_system/hemaia_system.srcs/sources_1/bd/hemaia_system/hemaia_system.bd]
export_ip_user_files -of_objects [get_files ./hemaia_system/hemaia_system.srcs/sources_1/bd/hemaia_system/hemaia_system.bd] -no_script -sync -force -quiet
create_ip_run [get_files -of_objects [get_fileset sources_1] ./hemaia_system/hemaia_system.srcs/sources_1/bd/hemaia_system/hemaia_system.bd]

# Re-add hemaia chip includes
set build hemaia_system

export_ip_user_files -of_objects [get_ips occamy_chip] -no_script -sync -force -quiet
eval [exec sed {s/current_fileset/get_filesets hemaia_system_occamy_chip_0/} define_defines_includes_no_simset.tcl]

# Do NOT insert BUFGs on high-fanout nets (e.g. reset). This will backfire during placement.
set_param logicopt.enableBUFGinsertHFN no

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
        set_property strategy Flow_AreaOptimized_high [get_runs $run]
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
    set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING false [get_runs $run]
    launch_runs $run -jobs ${nproc}
    wait_on_run $run
} else {
    puts "Skipping 100% complete run: $run"
}

# Create ILA. Attach all signals that were previously marked debug.
# For internal signals: Add "(* mark_debug = "true" *)" before signal definition in HDL code.
# For blockdesign-level signals: Use "set_property HDL_ATTRIBUTE.DEBUG $DEBUG [get_bd_nets ...]" in hemaia_system_vpk180_bd.tcl
if ($DEBUG) {
    open_run synth_1 -name synth_1
    # Create core
    puts "Creating debug core..."
    create_debug_core u_ila_0 ila
    set_property ALL_PROBE_SAME_MU true [get_debug_cores u_ila_0]
    set_property ALL_PROBE_SAME_MU_CNT 4 [get_debug_cores u_ila_0]
    set_property C_ADV_TRIGGER true [get_debug_cores u_ila_0]
    set_property C_DATA_DEPTH 4096 [get_debug_cores u_ila_0]
    set_property C_EN_STRG_QUAL true [get_debug_cores u_ila_0]
    set_property C_INPUT_PIPE_STAGES 0 [get_debug_cores u_ila_0]
    set_property C_TRIGIN_EN false [get_debug_cores u_ila_0]
    set_property C_TRIGOUT_EN false [get_debug_cores u_ila_0]

    ## Clock
    set_property port_width 1 [get_debug_ports u_ila_0/clk]
    connect_debug_port u_ila_0/clk [get_nets [list hemaia_system_i/versal_cips_0/inst/pl0_ref_clk]]

    set debugNets [lsort -dictionary [get_nets -hier -filter {MARK_DEBUG == 1}]]
    set netNameLast ""
    set probe_i 0
    # Loop through all nets (add extra list element to ensure last net is processed)
    foreach net [concat $debugNets {""}] {
        # Remove trailing array index
        regsub {\[[0-9]*\]$} $net {} netName
        # Create probe after all signals with the same name have been collected
        if {$netNameLast != $netName} {
            if {$netNameLast != ""} {
                puts "Creating probe $probe_i with width [llength $sigList] for signal '$netNameLast'"
                # probe0 already exists, and does not need to be created
                if {$probe_i != 0} {
                    create_debug_port u_ila_0 probe
                }
                set_property port_width [llength $sigList] [get_debug_ports u_ila_0/probe$probe_i]
                set_property PROBE_TYPE DATA_AND_TRIGGER [get_debug_ports u_ila_0/probe$probe_i]
                connect_debug_port u_ila_0/probe$probe_i [get_nets $sigList]
                incr probe_i
            }
            set sigList ""
        }
        lappend sigList $net
        set netNameLast $netName
    }

    set_property target_constrs_file hemaia_system/hemaia_system.srcs/constrs_1/imports/hemaia_system/hemaia_system_vpk180_impl.xdc [current_fileset -constrset]
    if { $EXT_JTAG } {
        set_property target_constrs_file hemaia_system/hemaia_system.srcs/constrs_1/imports/hemaia_system/hemaia_system_vpk180_impl_ext_jtag.xdc [current_fileset -constrset]
    }
    save_constraints -force

    implement_debug_core

    write_debug_probes -force probes.ltx
}

# Disable the error message for the ADEF-911 DRC before the implementation because we don't need to do the simulation in Vivado. 
set_property SEVERITY {Warning} [get_drc_checks ADEF-911]
set_property IS_ENABLED 0 [get_drc_checks {ADEF-911}]

# Implement
set_property strategy Congestion_SpreadLogic_high [get_runs impl_1]
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
