# XDMA byte-mask end-to-end regression

Run from the repository inside the locked build environment:

```sh
snax-shell
python target/snitch_cluster/sw/snax/xdma/tests/byte_mask_e2e/run.py
```

The test generates the complete production XDMATop with a **512-bit TCDM bank**,
a 32-bit CSR interface and matching software address definitions. It compiles
the real C driver and a C test program, and runs them against that RTL in
Verilator. A SystemVerilog SRAM honors the writer's actual byte strobes.

The C program configures, starts and waits for four transfers:

- All bytes enabled through the existing copy API.
- Only bytes 32 through 63 enabled through the second mask CSR.
- Different patterns in the low and high mask words.
- Both mask words cleared.

After each transfer it checks every destination byte, the unchanged source, and
guards before and after the destination. Source values differ from the initial
destination sentinel, so missing upper-byte writes cannot pass accidentally.

The host-side SNRT accessors issue real CSR bus transactions. They do not mock
register values or perform copies. Memory is updated by the simulated SRAM only
when the production XDMA writer issues a request with the corresponding strobe.
Host memory access is used for initialization and checking.

This is an end-to-end **C-driver/XDMA/memory** simulation. It does not execute a
RISC-V CPU or boot a complete Snitch cluster. The standard cluster generator
restricts the bank/core width to 32 or 64 bits, so this standalone harness exposes
the wider bank without changing production configuration or existing tests.

For a before/after check, point the same test at an isolated pre-fix checkout
with its Chisel submodule initialized:

```sh
python target/snitch_cluster/sw/snax/xdma/tests/byte_mask_e2e/run.py \
  --source-root /path/to/pre-fix-checkout \
  --build-dir /path/to/separate-build
```

The source root supplies both RTL and the C driver; the test and SRAM harness
stay identical. The original implementation should fail the first transfer at
destination byte 32: its mask contains only the low 32 enable bits. No newly
added driver function is required to reach that comparison.

Build products stay in the ignored build directory by default. Failures return a
nonzero status, and a cycle limit catches stalled configuration or completion.


Validated on 29 October 2026:

- Original commit 2cd09874: FAIL at byte 32, expected 0x20, received 0xa5.
  The transfer completed; the upper destination byte remained unchanged.
- Fixed commit 8cf08380: PASS for all four transfers (3749 simulation cycles).

The same generated-hardware configuration, C test, CSR bridge and SRAM harness
were used. Both the driver and RTL came from the respective source commit.


The separate .github/workflows/byte-mask-e2e.yml CI workflow runs this test and
the C-driver register-write tests on pull requests and pushes to main.
