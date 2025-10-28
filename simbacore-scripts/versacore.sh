# In SNAX container
make -C target/snitch_cluster CFG_OVERRIDE=cfg/snax_versacore_cluster.hjson rtl-gen
make -C target/snitch_cluster CFG_OVERRIDE=cfg/snax_versacore_cluster.hjson  SELECT_RUNTIME=rtl-generic SELECT_TOOLCHAIN=llvm-generic sw -j 
make -C target/snitch_cluster CFG_OVERRIDE=cfg/snax_versacore_cluster.hjson vsim_preparation


# In bash
make -C target/snitch_cluster/ CFG_OVERRIDE=cfg/snax_versacore_cluster.hjson bin/snitch_cluster.vsim > vsim.log 2>&1


./target/snitch_cluster/bin/snitch_cluster.vsim ./target/snitch_cluster/sw/apps/snax-versacore-matmul/build/snax-versacore-matmul.elf > vsim.log 2>&1
