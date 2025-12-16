# SimbaCore Workflow commands

## Prepare RTL

This is entirely managed by Robin.

- Make changes in chisel-ssm and push to git. This automatically updates SimbaCore.sv
- When architecture changes:
  - parse widths from SimbaCore.sv and update shell_wrapper accordingly
  - `python ./simbacore-scripts/update_simbacore_params.py`
  - Manually check the Streamer widths in config.hjson are still correct

## Build RTL/simulator

`bash scripts/build_sim.sh`

## Test

`bash scripts/regression_test.sh`

## Debug

`cd target/snitch_cluster`

[bash|target] Run SimbaCore test programs

- `bin/snitch_cluster.vsim sw/apps/nop/build/nop.elf | tee vsim.log`
- `bin/snitch_cluster.vsim sw/apps/snax-simbacore-main/build/snax-simbacore-main.elf | tee vsim.log`
- `bin/snitch_cluster.vsim sw/apps/snax-simbacore-osgemm/build/snax-simbacore-osgemm.elf | tee vsim.log`
- `bin/snitch_cluster.vsim sw/apps/snax-simbacore-isgemm/build/snax-simbacore-isgemm.elf | tee vsim.log`
- `bin/snitch_cluster.vsim sw/apps/snax-simbacore-simd/build/snax-simbacore-simd.elf | tee vsim.log`
- `bin/snitch_cluster.vsim sw/apps/snax-simbacore-main-full/build/snax-simbacore-main-full.elf | tee vsim.log`

[bash|target] Run SimbaCore test program in GUI (with VNC)

- `bin/snitch_cluster.vsim.gui sw/apps/snax-simbacore-main/build/snax-simbacore-main.elf`
- `bin/snitch_cluster.vsim.gui sw/apps/snax-simbacore-isgemm/build/snax-simbacore-isgemm.elf`

[snax|target] Make traces (from .dasm to .txt)

`make traces`
