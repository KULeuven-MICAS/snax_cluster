![CI](https://github.com/KULeuven-MICAS/snax_cluster/actions/workflows/ci.yml/badge.svg)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# SNAX Cluster

This repository hosts the hardware and base software for the SNAX cluster and its generator. SNAX is a high-efficiency compute cluster platform designed for multi-accelerator development. It provides a standard SNAX shell for conveniently integrating versatile accelerators. Moreover, it contains a variety of accelerators that are available for use.

The figure below shows an architectural overview of the SNAX cluster.

![image](https://github.com/KULeuven-MICAS/snax_cluster/assets/26665295/9242dd65-be3a-4472-8ae5-e026992f6a95)

SNAX supports several design-time and run-time configurations to support a variety of accelerators. For example, some design-time configurations include customizing the memory sizes, the interconnect structure connecting the accelerators to memory, the number of Snitch cores controlling accelerators, and so much more. For run-time configurations, we provide data streamers and reshufflers to aid accelerators in organizing data layouts in memory and managing data access patterns.

* SNAX is being co-developed with an MLIR-based toolchain, available at [KULeuven-MICAS/snax-mlir](https://github.com/KULeuven-MICAS/snax-mlir).
* SNAX clusters are also integrated in a CVA6 host system available at [KULeuven-MICAS/HeMAia](https://github.com/KULeuven-MICAS/HeMAiA)

## Getting Started

### Quick Start

As a quick start, it is recommended to download our pre-built docker container.

1 ) Clone the repository and go inside it after:

``` bash
git clone https://github.com/KULeuven-MICAS/snax_cluster.git --recurse-submodules
```

Note: If you cloned the repository already but without the recursed submodules, update the summodules inside the repository:

```bash
git submodule update --init --recursive
```

2 ) Download the docker:

``` bash
docker pull ghcr.io/kuleuven-micas/snax:main
```

3 ) Start the container:

```bash
podman run -it -v `pwd`:/repo -w /repo ghcr.io/kuleuven-micas/snax:main
```

Alternatively, you can use map the real path of your working `snax_cluster` directory:

```bash
podman run -it -v `pwd`:`pwd` -w `pwd` ghcr.io/kuleuven-micas/snax:main
```

4 ) Generate all RTL files of your chosen accelerator. For this quick start, we will begin with a simple ALU. Other accelerator configurations can be found under `snax_cluster/target/snitch_cluster/cfg/.`

```bash
make -C target/snitch_cluster CFG_OVERRIDE=cfg/snax_alu_cluster.hjson rtl-gen
```

5 ) Build the HW. The `-j` option parallelizes the build. The command below builds a Verilator binary for simulation.

```bash
make -C target/snitch_cluster CFG_OVERRIDE=cfg/snax_alu_cluster.hjson bin/snitch_cluster.vlt -j
```

6 ) Build the SW. The `-j` option parallelizes the build.

```bash
make -C target/snitch_cluster CFG_OVERRIDE=cfg/snax_alu_cluster.hjson sw -j
```

7 ) Run the built SW of the accelerator on the built Verilator binary. The `--vcd` is optional to dump a vcd file to be viewed with `gtkwave`. A `sim.vcd` file gets dump from where you ran the command.

```bash
./target/snitch_cluster/bin/snitch_cluster.vlt ./target/snitch_cluster/sw/apps/snax-alu/build/snax-alu.elf --vcd
```

You should see a log:

```bash
VCD wave generation enabled
[fesvr] Wrote 36 bytes of bootrom to 0x1000
[fesvr] Wrote entry point 0x80000000 to bootloader slot 0x1020
[fesvr] Wrote 72 bytes of bootdata to 0x1024
[Tracer] Logging Hart          0 to logs/trace_chip_00_hart_00000.dasm
[Tracer] Logging Hart          1 to logs/trace_chip_00_hart_00001.dasm
Accelerator Done! 
Accelerator Cycles: 25 
Number of errors: 0 
```

You can view the waveform with:

```bash
gtkwave sim.vcd
```


### Detailed Guides

There is a detailed [getting started guide](https://kuleuven-micas.github.io/snax_cluster/ug/getting_started.html) that contains several information about installation and directory setup.

We provide a [detailed tutorial](https://kuleuven-micas.github.io/snax_cluster/tutorial/introduction.html) on integrating a custom accelerator into the SNAX cluster platform. 

## Content

What can you expect to find in this repository?

- The RISC-V [Snitch integer core](https://pulp-platform.github.io/snitch_cluster/rm/snitch.html). This can be useful stand-alone if you are just interested in re-using the core for your project, e.g., as a tiny control core or you want to make a peripheral smart. The sky is the limit.
- The [SNAX cluster](https://kuleuven-micas.github.io/snax_cluster/tutorial/architectural_overview.html). A highly configurable cluster provides a standard and clean accelerator integration interface.

- A [CSR Manager](https://kuleuven-micas.github.io/snax_cluster/tutorial/csrman_design.html) is included to support a standardized control and management of registers through CSR instructions.
  
- The [Data Streamer](https://kuleuven-micas.github.io/snax_cluster/tutorial/streamer_design.html) streamlines the data access patterns for the accelerator.
- Versatile accelerator examples, including a [Chisel-based GEMM accelerator generator](https://github.com/KULeuven-MICAS/snax-gemm), a [rescale SIMD accelerator generator](https://github.com/KULeuven-MICAS/snax-postprocessing-simd), and a [Data Reshuffler](https://github.com/KULeuven-MICAS/snax_cluster/blob/main/hw/snax_data_reshuffler/doc/snax_data_reshuffler.md) for data layout transformation, are provided for reference.
- Software example applications for the SNAX cluster and corresponding accelerators.
- RTL simulation environments for Verilator, Questa Advanced Simulator, and VCS.

## Directory Structure

The project is organized as a monolithic repository. Both hardware and software are co-located. Standalone accelerators can have their own repositories and can be imported here using [Bender](https://github.com/pulp-platform/bender).

The file tree is visualized as follows:

```
├── hw
│   ├── chisel
│   │   ├── csr_manager
│   │   └── streamer
│   ├── snax_accelerator_1
│   ├── snax_accelerator_2
│   ├── snitch_stuff
│   └── templates
├── sw
├── target
│   └── snitch_cluster
│       ├── config
│       ├── generated
│       └── sw
│           ├── apps
│           │   ├── snax_system_1
│           │   └── snax_system_2
│           └── snax_lib 
│               ├── snax_system_1
│               └── snax_system_2
└── util
    ├── clustergen
    └── wrappergen
```

The top-level is structured as follows:

* `docs`: [Documentation](documentation.md) of the generator and software.
  Contains additional user guides.
* `hw`: All hardware IP components. The source files are either specified by SystemVerilog, Chisel, or a template to generate these files.
* `sw`: Hardware independent software, libraries, runtimes, etc.
* `target`: Contains the testbench setup, cluster configuration specific hardware and software, libraries, runtimes, etc.
* `util`: Utility and helper scripts.

# Publications

- **[OpenGeMM: A High-Utilization GeMM Accelerator Generator with Lightweight RISC-V Control and Tight Memory Coupling](https://arxiv.org/abs/2411.09543)**  
  Yi X, Antonio R, Dumoulin J, Sun J, Van Delm J, Paim G, Verhelst M  
  arXiv preprint arXiv:2411.09543. 2024 Nov 14.  

# Acknowledgement

SNAX is an extension of the original [Snitch Cluster](https://github.com/pulp-platform/snitch_cluster) framework. SNAX got its name as "SNitch Acceleration eXtension".

# Running snax-forge bundles

[snax-forge](https://github.com/KULeuven-MICAS/snax-forge) generates an accelerator for a design point and hands it over as a bundle, one generated accelerator and everything this repository builds it from:

- `snax_forge.hjson`: the cluster `snax_forge_cluster`, whose accelerator names its RTL's file list (`snax_forge_rtl: "rtl/files.f"`);
- `snax_forge_streamer.hjson`, `accelerator.json`: the streamer and the accelerator's description that the cluster file's `$ref`s name;
- `rtl/`: `files.f` and the SystemVerilog it lists (`snax_forge_shell_wrapper` with its CSRs, a controller per group and the datapaths; the per-group busy and done pins are left unconnected);
- the program: `main.c`, `data.h` and `include/` (the streamer's and the accelerator's CSR headers);
- `bundle.json`: the design point, the tag and the snax-forge commit;
- `forge_expected.txt`, in a bundle made by snax-forge's `pixi run forge`: the `forge counter` lines and the `forge data: 0 errors` line of the bundle's own SNAX-GVSoC run, which a run on this repository's RTL must print too.

snax-forge's bundles are no longer kept or checked in this repository's CI: they are checked locally, on vsim, before a snax-forge change to the emitted RTL or the program merges. `target/snitch_cluster/snax_forge.mk` runs them with this repository's own targets:

| | |
|---|---|
| `CFG_OVERRIDE` | the bundle's cluster file, by its absolute path |
| `FORGE` | the program's folder; default the folder of `CFG_OVERRIDE` |
| `ELF` | an ELF to run, such as the one snax-forge built from the bundle; unset, the program built from `FORGE` |
| `SIM` | `vlt` (default) or `vsim` |
| `snax-forge-sim` | builds `bin/snitch_cluster.$(SIM)`; for `vsim` after `vsim_preparation` |
| `snax-forge-sw` | builds `sw/apps/snax-forge`'s ELF from `FORGE` (the ELF only), after checking its `include/streamer_csr_addr_map.h` against snaxgen's (the `#define` lines) |
| `snax-forge-run` | runs the ELF, writes its output to `logs/snax-forge-$(SIM).txt` beside the traces, and fails on the program's nonzero exit code |
| `snax-forge-check` | runs each bundle of `BUNDLES` (absolute folders) in order with `SIM` and the development image `SNAX_CLUSTER_IMAGE`, and compares its `forge counter` and `forge data` lines with the bundle's `forge_expected.txt`: `PASS` or `FAIL` per bundle in `logs/snax-forge-check.txt` (with each run's `forge cycles` line), each bundle's output in `logs/snax-forge-<name>-$(SIM).txt`; it fails when a bundle failed |

The local check: in snax-forge, `pixi run forge-set quick` (or `full`) makes a set's bundles and prints one line; here, from `target/snitch_cluster`, run that line with the image:

```bash
make snax-forge-check BUNDLES="/abs/snax-forge/out/bundle/vecadd_i32 /abs/snax-forge/out/bundle/dot_4" SIM=vsim SNAX_CLUSTER_IMAGE=c5db84ad7bce
```

```
PASS vecadd_i32
PASS dot_4
```

With `SIM=vsim`, `rtl-gen` and `vsim_preparation` run in the development image (`c5db84ad7bce` on MICAS's servers), under podman with this checkout and the bundle mounted at their own paths, and the build and the run outside it, with `bender` (this repository's pixi environment's, when not on `PATH`) and Questa on `PATH`; with `SIM=vlt` every step runs in the image. Questa exits 0 whatever the program returns, so `snax-forge-run` reads the exit code from the testbench's `[SUCCESS] Program finished successfully` line.

One bundle by hand, as snax-forge's `pixi run forge` prints its commands (`B` the bundle's cluster file, `E` the ELF snax-forge built from it):

```bash
# in the image, from snax_cluster
make -C target/snitch_cluster clean-generated clean-vlt clean-vsim
make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=$B
make -C target/snitch_cluster vsim_preparation CFG_OVERRIDE=$B
# outside it, bender and Questa on PATH
make -C target/snitch_cluster snax-forge-sim SIM=vsim CFG_OVERRIDE=$B
make -C target/snitch_cluster snax-forge-run SIM=vsim CFG_OVERRIDE=$B ELF=$E
```

To make a bundle, see snax-forge's README, [Getting started](https://github.com/KULeuven-MICAS/snax-forge#getting-started).
