# Native wide XDMA access to TCDM

The optional cluster setting **xdma_use_wide_tcdm** connects XDMA to the
existing wide memory interconnect. It defaults to false.

With 64-bit physical banks and dma_data_width=512, default mode has eight
64-bit reader channels and eight writer channels. Wide mode has one 512-bit
reader and one writer. The CPU/LSU, narrow fabric and physical SRAMs stay
64 bits wide. The existing wide/narrow mux slices a wide request into eight
bank requests and slices all 64 byte enables correspondingly.

## Routing and compatibility

The wide interconnect has three inputs in wide mode: existing AXI/iDMA,
XDMA reader and XDMA writer. It arbitrates per destination superbank and
retains existing fixed-latency response ownership tracking. The selected
wide superbank blocks its narrow ports for that access. Zero strobes keep
bytes unchanged but do not cancel the corresponding bank requests.

The 16 legacy narrow XDMA slots remain allocated and inactive in wide mode.
Other accelerator port indices and the sparse topology therefore remain
stable. Existing configurations keep their original routing.

The supplied configurations exercise both narrow fabrics:

* **snax_xdma_wide_cluster.hjson** selects the sparse interconnect.
* **snax_xdma_wide_cluster_old.hjson** selects the old logarithmic interconnect,
  following the switch used by snax_alu_cluster_old.hjson.

In both cases, CPU traffic uses the selected narrow fabric. Native XDMA and
iDMA use the shared 512-bit interconnect and meet CPU traffic at the existing
wide/narrow bank muxes. The sparse fabric does not split these native XDMA
requests into independently arbitrated 64-bit channels. The mux slices each
512-bit access into eight 64-bit bank accesses with their byte enables.
Neither configuration changes physical bank width or the CPU/LSU interface.

Wide mode is selected at RTL generation time. Each request must be aligned
to the complete wide word (64 bytes for the supplied configuration); an RTL
assertion checks this. It is not a runtime mode switch or a general adapter
for arbitrary eight-channel spatial access patterns.

The logical XDMA port width is also used by the Chisel configuration format
for channel/mask fields and length/stride units. The generated SV wrapper
passes the same logical width as WordlineWidth to the existing AXI adapter.
The physical bank word remains independent. Mixed-width remote peers are
not covered by this local regression and must not be assumed compatible.

## Build and run

Run from the repository root in the locked Pixi environment. When switching
an already-built checkout between configurations, run clean-generated first
so the generated Bender target selection matches the new configuration.
The XDMA driver depends on its generated CSR header, and applications depend
on the linked driver object, so a changed register layout rebuilds both.

    make -C target/snitch_cluster clean-generated
    make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=cfg/snax_xdma_wide_cluster.hjson
    make -C target/snitch_cluster bin/snitch_cluster.vlt CFG_OVERRIDE=cfg/snax_xdma_wide_cluster.hjson -j4
    make -C target/snitch_cluster sw CFG_OVERRIDE=cfg/snax_xdma_wide_cluster.hjson -j4
    cd target/snitch_cluster
    ./run.py --simulator verilator sw/runtime.yaml sw/snax-xdma-wide-run.yaml -j4

From the repository root, repeat with CFG_OVERRIDE=cfg/snax_xdma_wide_cluster_old.hjson
to test the logarithmic narrow fabric. These alternative configurations reuse
the generated module names and must be built and run separately.

The existing ci.yml XDMA job builds the normal configuration and both wide
configurations. Each wide job runs the same runtime, byte-mask and contention
tests, so both narrow fabrics are exercised alongside the native wide path.
No standalone generated-RTL test lives in the software-library directory.

## What the cluster tests check

* **snax-xdma-byte-mask** runs the existing six transfers plus upper-word-only
  and lower-word-only cases when multiple mask CSRs are generated. It checks
  each destination byte, destination guards, and unchanged source data.
* **snax-xdma-wide-contention** requires one 512-bit channel and two mask CSRs
  at compile time. It launches original iDMA and XDMA copies together while
  the CPU accesses separate TCDM scratch space. Four mask patterns check both
  destinations, unchanged sources and guards across all bank groups.
  This is a functional contention test, not a throughput or fairness bound.
* **util/snaxgen/tests** checks default compatibility, invalid bank groupings,
  and the actual generated wrapper's local and metadata widths.

CSR0=0 and CSR1=FFFFFFFF must preserve destination bytes 0..31 of each wide
word and copy bytes 32..63. Every XDMA channel stays enabled: the result must
come from byte enables rather than disabling whole channels.

The unchanged default configuration must also pass its existing runtime,
XDMA and SIMD regressions. Build simulator and software for the same
configuration: generated header layouts differ between modes.
The two wide configurations intentionally share the same geometry and CSR map;
only the narrow topology and corresponding Bender target selection differ.
