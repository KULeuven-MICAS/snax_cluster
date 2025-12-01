#!/bin/bash
# SimbaCore Workflow Script

set -e


#####################
# Step 0: Prepare RTL
#####################

# This is entirely managed by Robin. 

# Make changes in chisel-ssm and push to git. This automatically updates SimbaCore.sv
# When architecture changes: parse widths from SimbaCore.sv and update shell_wrapper accordingly
python ./simbacore-scripts/update_simbacore_params.py
# Manually check the Streamer widths in config.hjson are still correct 

###################################
# Step 1: Generate RTL and wrappers
###################################

# [bash] Make sure bender is up-to-date
bender update --fetch

# Start the container (denoted as [snax] in the script):
podman run -it -v `pwd`:`pwd` -w `pwd` ghcr.io/kuleuven-micas/snax:main
# [...|target] denotes running from this directory
cd target/snitch_cluster

# [snax|target] Generate RTL and wrappers 
make CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson rtl-gen


########################
# Step 2: Run QuestaSim
########################

# When adding extra external sv sources, add to bender.yaml and run `bender update`
# Then, remove the vsim work folder and run this
make CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson vsim_preparation

# [snax|target] Build the software
make CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson sw -j 

# [bash|target] Create QuestaSim binary and run programs
make CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson bin/snitch_cluster.vsim 

# [bash|target] Run SimbaCore test programs
bin/snitch_cluster.vsim sw/apps/nop/build/nop.elf | tee vsim.log 
bin/snitch_cluster.vsim sw/apps/snax-simbacore-main/build/snax-simbacore-main.elf | tee vsim.log
bin/snitch_cluster.vsim sw/apps/snax-simbacore-osgemm/build/snax-simbacore-osgemm.elf | tee vsim.log
bin/snitch_cluster.vsim sw/apps/snax-simbacore-isgemm/build/snax-simbacore-isgemm.elf | tee vsim.log

# Debug
# [bash|target] Run SimbaCore test program in GUI (with VNC)
bin/snitch_cluster.vsim.gui sw/apps/snax-simbacore-main/build/snax-simbacore-main.elf 
bin/snitch_cluster.vsim.gui sw/apps/snax-simbacore-isgemm/build/snax-simbacore-isgemm.elf 

# [snax|target] Make traces (from .dasm to .txt)
make traces

###########################
# Step 3 Elaborate design
###########################

# [bash|root] Generate flist from bender
bender script synopsys -t synthesis -t snax_simbacore -t snax_simbacore_cluster  > scripts_generated/flist.tcl

