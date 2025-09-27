set system_top snax_simbacore_cluster_wrapper

#-----------------------------
# Setting up multiple cores
#-----------------------------

# Ensure multi-core operation
set_host_options -max_cores 16
set disable_multicore_resource_checks true
set dcnxt_adaptive_multithreading true


#-----------------------------
# Defines the work path
#-----------------------------
set workdir "."
define_design_lib WORK -path $workdir/work

#-----------------------------
# Source the filelist
#-----------------------------

source $workdir/flist.tcl
# Checks the synthesizability of the design and for latches
elaborate $system_top
current_design $system_top


quit
