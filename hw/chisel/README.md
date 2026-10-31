# SNAX Framework Chisel components
This directory contains SNAX framework components developed in Chisel. You can find documentation for these modules here:

### Streamer
[documentation](doc/streamer.md)

### ReqRsp Manager
TODO

### XDMA
[documentation](doc/xdma.md)

### Unit tests
Run the tests with `sbt test` here or in `hw/chisel_acc`.
They write no waveforms by default, which keeps the Verilator builds fast.
To get a VCD for each test in `test_run_dir/`, set `SNAX_VCD=1`, for example `SNAX_VCD=1 sbt "testOnly snax.utils.BasicCounterTester"`.
The randomized tests in `hw/chisel_acc` use a fixed seed, so every run is the same.
Each of them prints its seed as `SNAX_TEST_SEED=<n>`; set `SNAX_TEST_SEED` to that value to re-run a CI failure locally, or to another number to try other inputs.
