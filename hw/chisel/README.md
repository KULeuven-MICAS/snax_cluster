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
