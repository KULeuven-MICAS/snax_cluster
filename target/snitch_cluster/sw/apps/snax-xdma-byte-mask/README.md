# XDMA byte-mask cluster test

This RISC-V application uses the production XDMA C driver and cluster runtime.
The DM core initializes separate source and destination buffers in TCDM,
configures XDMA, starts each transfer, waits for completion, and checks memory.

The six cases check the scalar full-mask copy, a full mask set through the word
array API, distinct adjacent mask words, only the highest byte of each bank word,
a cleared mask, and a scalar full-mask copy after clearing. Every transfer checks
each destination byte, unchanged source data, and guards around the destination.
The generated `XDMA_WIDTH`, `XDMA_SPATIAL_CHAN`, and
`XDMA_DST_ENABLED_BYTE_CSR_NUM` definitions determine the buffer and mask sizes.

The existing KUL and XDMA clusters use 64-bit banks, which need only one mask CSR.
On those clusters this app checks the C driver, CSR interface, and memory writes
through the complete compute cluster; it does not reproduce the original wide
bank failure. The Scala byte-mask tests cover allocation, mapping, word ordering,
and byte strobes for wide banks. A 512-bit aggregate beat with eight 64-bit
channels is not a 512-bit bank.

From the repository root, inside the locked Pixi environment:

```sh
snax-shell
make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=cfg/snax_xdma_cluster.hjson
make -C target/snitch_cluster bin/snitch_cluster.vlt CFG_OVERRIDE=cfg/snax_xdma_cluster.hjson
make -C target/snitch_cluster sw CFG_OVERRIDE=cfg/snax_xdma_cluster.hjson
cd target/snitch_cluster
./bin/snitch_cluster.vlt sw/apps/snax-xdma-byte-mask/build/snax-xdma-byte-mask.elf
```

A successful run prints `[ByteMask] PASS all six transfers` and exits with status
zero. Any data or guard mismatch prints the failing case and returns an error.
The app is built for `cfg/snax_KUL_cluster.hjson` as well. Use a simulator and
generated software definitions built for the same cluster configuration.
