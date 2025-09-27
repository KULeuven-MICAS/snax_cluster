# SimbaCore workflow

**Step 1**: Generate SimbaCore.sv in `chisel-ssm`

**Step 2**: Parse widths from SimbaCore.sv and update shell_wrapper accordingly

- `python ./simbacore-work/update_simbacore_params.py`

**Step 3**: Make sure the Streamer widths in config.hjson are still correct

**Step 4**: Generate RTL & wrappers in SNAX environment

- `podman run -it -v `pwd`:`pwd`-w`pwd` ghcr.io/kuleuven-micas/snax:main`

- `make -C target/snitch_cluster CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson rtl-gen`

**Step 5** (optional): Verify port connections by running QuestaSim

In container:

- `make -C target/snitch_cluster CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson vsim_preparation`
- `make CFG_OVERRIDE=cfg//snax_simba_cluster.hjson sw -j`

In bash:

- `make -C target/snitch_cluster/ CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson bin/snitch_cluster.vsim`
  - This creates a QuestaSim binary.
- `target/snitch_cluster/bin/snitch_cluster.vsim target/snitch_cluster/sw/apps/nop/build/nop.elf > vsim.log`
  - Runs a no-op program

**Step 6**: Generate flist from bender

- `bender script synopsys -t synthesis -t snax_simbacore -t snax_simbacore_cluster > simbacore-work/flist.tcl`

**Step 7**: Run Design Compiler (e.g., check with elaborate)

- `cd simbacore-work && dcnxt_shell -64bit -f dc.tcl > dc.log`
