# snax-dma-transpose

The DMA alone transposes an N × N matrix (N = 64) in L1 by 2-D transfers of one element per burst, the copy snax-forge's `data_transpose` makes for mvt's second half. It runs on any cluster with a DMA core; snax-forge's numbers come from `cfg/snax_snitch_cluster_1.hjson`, which has forge's L1 (128 KiB in 32 banks) and DMA (512 bit, 3-deep request FIFO) without an accelerator.

For each column k of A, one `snrt_dma_start_2d(T + p·k, A + b·k, b, b, p_A, N)`: N bursts of b bytes, A's side stepping its row pitch p_A, T's side stepping b. Each case runs twice and keeps the second run, so instruction cache refills stay out of the numbers.

| Case | Copies |
|---|---|
| `i8 col l1l1 p64` | one column of an int8 A in L1 (pitch 64 B): 64 bursts |
| `i8 l1l1 p64` | the whole int8 A in L1, rows contiguous: 4096 bursts |
| `i8 l1l1 p72` | the same with 72-byte rows (9 words, as snax-forge's `padded_rows`) |
| `i64 l1l1 p512` | an int64 A in L1: 4096 bursts of 8 B |
| `i8 l2l1 p64` | the int8 A read from DRAM, T in L1 |

| Field | Meaning |
|---|---|
| `bursts` | the 2-D transfers' bursts: N per column |
| `issue` | `mcycle` from before the first start to after the last: the core waits whenever the DMA's request FIFO is full |
| `done` | `mcycle` from before the first start to after `snrt_dma_wait_all` |
| `busy`, `r`, `w` | `DMA_BUSY`, `DMA_R_DONE`, `DMA_W_DONE` |
| `ar_stall`, `w_stall`, `r_stall` | the DMA's stall counters, as in snax-gvsoc's `dma-timing` |
| `busy/burst` | `busy` over `bursts` |
| `err` | elements of T that differ from A's transpose (A read in L1) |

The program returns the errors. `data/data.h` is written from `data/datagen.py` (seed 0) on the first build.

## Build and run

In the development image, from `target/snitch_cluster`, after building the runtime for the config (`make clean-sw && make sw CFG_OVERRIDE=cfg/snax_snitch_cluster_1.hjson`; a program links the last config's runtime):

    make -C sw/apps/snax-dma-transpose CFG_OVERRIDE=cfg/snax_snitch_cluster_1.hjson

The ELF is `sw/apps/snax-dma-transpose/build/snax-dma-transpose.elf`; it runs on `bin/snitch_cluster.vsim` built for the same config, as the `snitch-forge` programs do.
