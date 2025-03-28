# Review of Architecture

Previously we discussed the [Accelerator Design](./accelerator_design.md), the [CSR Manager Design](./csrman_design.md) and the [Streamer Design](./streamer_design.md). These are the main components that are tightly coupled together in a shell before connecting to the system. The figure below shows the system of interest again:

![image](https://github.com/KULeuven-MICAS/snitch_cluster/assets/26665295/0090e1a4-15c5-4852-a351-7c3ee0368e1f)

There are key wrappers: **(1) The accelerator shell** that needs to comply with the CSR manager and streamer interface. **(2) The CSR manager wrapper** and **(3) streamer wrapper** that converts the Chisel's unpacked signals to packed signals to connect RTL designs developed in SystemVerilog. An **(4) accelerator wrapper** that encapsulates all components together. Finally, the **(5) top-level cluster wrapper** connects the accelerator wrapper to the entire system. Components (2), (3), (4), and (5) are all generated through a script. There are more details in the [Wrapper Generation Script](#wrapper-generation-script) section about the wrapper generation.
# Step-by-step Guide to Build

Let's review what we have done so far to prepare the SNAX ALU system:

- Building your accelerator shell.
- Setting up the configuration file to our desired system requirements.
- Configuring the SNAX CSR manager.
- Configuring the SNAX streamer.

With this in place, we only need to make sure to include the RTL files needed for the build and do the build process. In this tutorial, we have several parts in place but we'll guide you through which components need to be modified. 

## Adding RTL Files

All RTL file lists are generated through [bender](https://github.com/pulp-platform/bender). Bender is a file management tool that functions like git submodules but with more features like file list generation in different forms. 

We first need to make sure we add all the files we need for our SNAX ALU. At the root of the repository, you should find a `Bender.yml` file. You will see several files related to the cluster. Scroll down to almost the bottom of the file and look for the `SNAX Accelerators` section. You should also find the SNAX ALU in there:

```yml
- target: snax-alu
    files:
      # Level 0
      - hw/snax_alu/src/snax_alu_csr.sv
      - hw/snax_alu/src/snax_alu_pe.sv
      - target/snitch_cluster/generated/snax_alu/snax_alu_csrman_CsrManager.sv
      - target/snitch_cluster/generated/snax_alu/snax_alu_streamer_StreamerTop.sv
      # Level 1
      - hw/snax_alu/src/snax_alu_shell_wrapper.sv
      - target/snitch_cluster/generated/snax_alu/snax_alu_csrman_wrapper.sv
      - target/snitch_cluster/generated/snax_alu/snax_alu_streamer_wrapper.sv
      # Level 2
      - target/snitch_cluster/generated/snax_alu/snax_alu_wrapper.sv
```

The `-target` parameter is a switch to include the files listed below it. Here, we use `snax-alu` as the target. Take note of this as we will use this definition in the makefile build later.

The files are listed according to a hierarchical structure from the base modules up to the system modules. This means that Level 0 contains modules that are not dependent on any other modules. Level 1 is dependent on Level 0 modules. Likewise, Level 2 is dependent on both Level 1 and Level 0 modules.

At Level 0, you will notice that we declared the `snax_alu_csr.sv` and `snax_alu_pe.sv` which are the base modules of our main SNAX ALU accelerator. It is important to declare the generated files which are the `snax_alu_csrman_CsrManager.sv` and `snax_alu_streamer_StreamerTop.sv`.

At Level 1, the `snax_alu_shell_wrapper.sv` is dependent on the ALU CSR and PEs and hence it is listed above the two. The wrappers for the CSR manager and streamer are also dependent on the Chisel-generated modules. Finally, Level 2 is the accelerator wrapper which is directly connected to the cluster system.

When you make your own accelerator you need to declare all the RTL files in this format too. Make sure it's under the SNAX Accelerator section just to make sure you don't mess up the structure.

### What About the Naming of the Generated Files?

The names of the generated files are dependent on the configuration file you set. If you recall in the `snax_acc_cfg`, there is a `snax_acc_name` parameter. Moreover, to match the bender target name specified earlier, you need to add the `bender_target: ["snax_alu"]` option too.

```hjson
snax_acc_cfg: {
    snax_acc_name: "snax_alu",
    bender_target: ["snax_alu"],
    snax_tcdm_ports: 16,
    snax_num_rw_csr: 3,
    snax_num_ro_csr: 2,
    snax_streamer_cfg: {$ref: "#/snax_alu_streamer_template" }
},
```
This is prefixed at the beginning of each generated file. Moreover, it also sets the generated directory for each accelerator you add. Such that we have:

```yml
- target: snax-alu
    files:
      # Level 0
      # ... Add some other files here
      - target/snitch_cluster/generated/${snax_acc_name}/${snax_acc_name}_csrman_CsrManager.sv
      - target/snitch_cluster/generated/${snax_acc_name}/${snax_acc_name}_streamer_StreamerTop.sv
      # Level 1
      # ... Add some other files here
      - target/snitch_cluster/generated/${snax_acc_name}/${snax_acc_name}_streamer_wrapper.sv
      # Level 2
      - target/snitch_cluster/generated/${snax_acc_name}/${snax_acc_name}_wrapper.sv
```

Where you should substitute `${snax_acc_name}`  with the parameter name you specified.

!!! note

    When you write the filenames for the `Bender.yml` file, make sure to have the correct top module name of the shell. For example, the `snax_alu_shell_wrapper.sv` should be the top-shell name of your accelerator. Then, the generated Chisel files should also haev the `${snax_acc_namee}` pre-fixed accordingly. Just like it was mentioned above.

## Build Process

1 - From the root of the repo navigate to `./target/snitch_cluster/`

2 - Generate the RTL files

```bash
make CFG_OVERRIDE=cfg/snax_alu_cluster.hjson rtl-gen
```

3 - Build the system:

```bash
make CFG_OVERRIDE=cfg/snax_alu_cluster.hjson bin/snitch_cluster.vlt -j
```

4 - Wait 5 to 10 minutes until the build finishes.

5 - Success!!!

That was easy, right?

# Seeing the Generated RTL Files

You can do step 2 whithout having to build the system yet. This becomes especially useful for designers to do a manual (eye-ball) sanity check on the connections.

```bash
make CFG_OVERRIDE=cfg/snax_alu_cluster.hjson rtl-gen
```

Generates all RTL files including the wrappers and the Chisel-generated CSR manager, streamers, (and possibly accelerators if need be).

# (Optional) Wrapper Generation Scripts

The original [Snitch Platform](https://github.com/pulp-platform/snitch_cluster) uses templated generation Python scripts. First, you have the accelerator wrapper script (`./util/snaxgen/snaxgen.py`) and then the cluster generation script (`./util/clustergen/cluster.py`).

In principle, both scripts use a specified configuration file like the `snax_alu_cluster.hjson` and load those configurations in a templated file `*.tpl`. Once loaded, it generates the target files like System Verilog designs or Chisel parameters.

## Accelerator Wrapper Script

The **accelerator wrapper** script uses any of the configuration files under the `./target/snitch_cluster/cfg/.` directory. Then load them in wrapper templates found in `./hw/templates/.` directory.

Inside the template directory we have:

- `snax_csrman_wrapper.sv.tpl` - is for the CSR manager wrapper. This is (2) from the figure above.
- `snax_streamer_wrapper.sv.tpl` - is for the streamer wrapper. This is (3) from the figure above.
- `snax_acc_wrapper.sv.tpl` - is for the accelerator wrapper. This is (4) from the figure above.
- `csrman_param_gen.scala.tpl` - this is a parameter file that Chisel uses for generating the CSR manager.
- `stream_param_gen.scala.tpl` - this is a parameter file that Chisel uses for generating the streamer.

We have shown previously how to use the `snaxgen` script. If you are working in Codespaces:

```bash
/workspaces/snax_cluster/util/snaxgen/snaxgen.py --cfg_path="/workspaces/snax_cluster/target/snitch_cluster/cfg/snax_alu_cluster.hjson" --tpl_path="/workspaces/snax_cluster/hw/templates/" --chisel_path="/workspaces/snax_cluster/hw/chisel/" --gen_path="/workspaces/snax_cluster/target/snitch_cluster/generated/"
```

If you are working in a docker container:

```bash
/repo/util/snaxgen/snaxgen.py --cfg_path="/repo/target/snitch_cluster/cfg/snax_alu_cluster.hjson" --tpl_path="/repo/hw/templates/" --chisel_path="/repo/hw/chisel/" --gen_path="/repo/target/snitch_cluster/generated/"
```

Where:
- `--cfg_path` - points to the configuration file of interest.
- `--tpl_path` - points to the template directory.
- `--chisel_path` - points to the chisel directory.
- `--gen_path` - points to the directory where all the generated files will be placed.

## Cluster Wrapper Script

The **cluster wrapper** made by the Snitch platform also does the same mechanism but is tailored towards the system integration. The main template is in `./hw/snitch_cluster/src/snitch_cluster_wrapper.sv.tpl`.

If you are working in Codespaces run the `clustergen` with:

```bash
/workspaces/snax_cluster/util/clustergen.py -c /workspaces/snax_cluster/target/snitch_cluster/cfg/snax_alu_cluster.hjson -o /workspaces/snax_cluster/target/snitch_cluster/generated --wrapper
```

If you are working in the container do:

```bash
/repo/util/clustergen.py -c cfg/snax_alu_cluster.hjson -o /repo/target/snitch_cluster/generated --wrapper
```

Where:
- `-c` - points to the target configuration file.
- `-o` - points to the directory where all the generated files will be placed.

The `clustergen` generates the `snitch_cluster_wrapper.sv` inside the `./target/snitch_cluster/generated` directory. You can see this after the build finishes.
# (Optional) More Details on the `snax_alu_cluster.hjson`

Let's go through the `snax_alu_cluster.hjson` file again and look at important configurations. We'll only discuss the notable ones that would be most useful for an accelerator designer. Remember, you can locate this at `./target/snitch_cluster/cfg/snax_alu_cluster.hjson` starting from the root of the repository.

The main `cluster` dictionary configures the entire system. Particularly, it affects the cluster wrapper which you can see in `./hw/snitch_cluster/src/snitch_cluster_wrapper.sv.tpl`. The file has several things in it but we'll just give you an overview of the few parameters.

First, under the cluster dictionary you have the TCDM `addr_width` and `data_width` definitions:

```hjson
addr_width: 48,
data_width: 64,
```
The `data_width` affects the memory bank data width sizes. By default, it's 64 already. Make sure this is consistent with the streamer definition.

The TCDM size is declared under the `tcdm` parameter. Where `size` is in kB and `banks` is the number of memory banks.

```hjson
tcdm: {
    size: 128,
    banks: 32,
},
```

Then we have DMA parameters for the built-in Snitch DMA. The `dma_data_width` affects the DMA bandwidth and also the wide interconnect of the cluster. This is a special parameter that can be useful if you want to hook your accelerator towards the wide interconnect. The FIFOs are for handling buffering inside the DMA.

```hjson
dma_data_width: 512,
dma_axi_req_fifo_depth: 3,
dma_req_fifo_depth: 3,
```

The `hives` contain information about the shared instruction cache and the cores we generate. This part is something we configure more often when we add more accelerators. Under the hives we have the instruction cache where `size` is in kB, `sets` refer to the *n-way* set associative, and `cacheline` is one word line.

```hjson
icache: {
    size: 8,
    sets: 2,
    cacheline: 256 
},
```

The `cores` refer to the core templates. We already saw an example of the `snax_alu_core_template`. A core template describes the functionalities of the Snitch core attached to an accelerator (or no accelerator). With the current list of cores we have 1 SNAX core attached to the SNAX ALU and 1 DMA core (a Snitch connected to the DMA only).

```hjson
cores: [
    { $ref: "#/snax_alu_core_template" },
    { $ref: "#/dma_core_template" },
]
```

For a multi-accelerator system attached to one cluster, we can have a cluster configuration like the one below. This time it has 3 other accelerators within one cluster.

```hjson
cores: [
    { $ref: "#/snax_data_reshuffler_template" },
    { $ref: "#/snax_streamer_simd_template" },
    { $ref: "#/snax_streamer_gemm_template" },
    { $ref: "#/dma_core_template" },
]
```

!!! note

    It is imperative that the last core should always be the DMA core. Otherwise, the generation scripts will break and you have no way of getting data into the TCDM memory.

## Some Exercises !!!

<details>
  <summary> What does the `-target` of the Bender.yml file do? </summary>
  It's used to specify if the components under the target will be included in the filelist generation.
</details>

<details>
  <summary> What do you need to update in the Makefile for a new accelerator? </summary>
  Just add the condition of the new accelerator configuration and make sure to add the appropriate -t tags. Make sure to include the `SNAX_GEN` path so that it creates the correct directory too.
</details>
