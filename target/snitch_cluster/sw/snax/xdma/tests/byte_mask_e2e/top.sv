// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

module byte_mask_e2e_top (
  input logic clk,
  input logic reset,
  input logic [31:0] csr_addr,
  input logic csr_write,
  input logic [31:0] csr_data,
  input logic [3:0] csr_strb,
  input logic csr_valid,
  output logic csr_ready,
  input logic rsp_ready,
  output logic rsp_valid,
  output logic [31:0] rsp_data,
  input logic [16:0] host_addr,
  input logic [7:0] host_data,
  input logic host_write,
  output logic [7:0] host_result
);

  logic [511:0] memory [0:2047];
  logic read_valid, read_write, read_rsp_valid;
  logic [16:0] read_addr;
  logic [511:0] read_rsp_data;
  logic write_valid, write_enable;
  logic [16:0] write_addr;
  logic [511:0] write_data;
  logic [63:0] write_strb;

  assign host_result = memory[host_addr[16:6]][8 * int'(host_addr[5:0]) +: 8];

  // One-cycle read response and byte-enabled writes, driven by actual XDMA requests.
  // No expected mask from the C test is used to update this memory.
  always_ff @(posedge clk) begin
    if (reset) begin
      read_rsp_valid <= 1'b0;
      read_rsp_data <= '0;
    end else begin
      read_rsp_valid <= read_valid;
      if (read_valid) begin
        assert (!read_write && read_addr[5:0] == 0)
          else $fatal(1, "Invalid read request");
        read_rsp_data <= memory[read_addr[16:6]];
      end
      if (write_valid) begin
        assert (write_enable && write_addr[5:0] == 0)
          else $fatal(1, "Invalid write request");
        for (int i = 0; i < 64; i++) begin
          if (write_strb[i])
            memory[write_addr[16:6]][8*i +: 8] <= write_data[8*i +: 8];
        end
      end
      if (host_write) begin
        assert (!write_valid) else $fatal(1, "Host initialization overlaps a DMA write");
        memory[host_addr[16:6]][8 * int'(host_addr[5:0]) +: 8] <= host_data;
      end
    end
  end

  byte_mask_e2e_xdma dut (
    .clock(clk),
    .reset(reset),
    .io_clusterBaseAddress(48'b0),
    .io_csrIO_req_ready(csr_ready),
    .io_csrIO_req_valid(csr_valid),
    .io_csrIO_req_bits_addr(csr_addr),
    .io_csrIO_req_bits_write(csr_write),
    .io_csrIO_req_bits_data(csr_data),
    .io_csrIO_req_bits_strb(csr_strb),
    .io_csrIO_rsp_ready(rsp_ready),
    .io_csrIO_rsp_valid(rsp_valid),
    .io_csrIO_rsp_bits_data(rsp_data),
    .io_remoteXDMACfg_fromRemote_ready(),
    .io_remoteXDMACfg_fromRemote_valid(1'b0),
    .io_remoteXDMACfg_fromRemote_bits(512'b0),
    .io_remoteXDMACfg_toRemote_ready(1'b1),
    .io_remoteXDMACfg_toRemote_valid(),
    .io_remoteXDMACfg_toRemote_bits(),
    .io_tcdmReader_req_0_ready(1'b1),
    .io_tcdmReader_req_0_valid(read_valid),
    .io_tcdmReader_req_0_bits_addr(read_addr),
    .io_tcdmReader_req_0_bits_write(read_write),
    .io_tcdmReader_req_0_bits_data(),
    .io_tcdmReader_req_0_bits_strb(),
    .io_tcdmReader_req_0_bits_priority(),
    .io_tcdmReader_rsp_0_valid(read_rsp_valid),
    .io_tcdmReader_rsp_0_bits_data(read_rsp_data),
    .io_tcdmWriter_req_0_ready(1'b1),
    .io_tcdmWriter_req_0_valid(write_valid),
    .io_tcdmWriter_req_0_bits_addr(write_addr),
    .io_tcdmWriter_req_0_bits_write(write_enable),
    .io_tcdmWriter_req_0_bits_data(write_data),
    .io_tcdmWriter_req_0_bits_strb(write_strb),
    .io_tcdmWriter_req_0_bits_priority(),
    .io_remoteXDMAData_fromRemote_ready(),
    .io_remoteXDMAData_fromRemote_valid(1'b0),
    .io_remoteXDMAData_fromRemote_bits(512'b0),
    .io_remoteXDMAData_fromRemoteAccompaniedCfg_readyToTransfer(),
    .io_remoteXDMAData_fromRemoteAccompaniedCfg_taskID(),
    .io_remoteXDMAData_fromRemoteAccompaniedCfg_length(),
    .io_remoteXDMAData_fromRemoteAccompaniedCfg_taskType(),
    .io_remoteXDMAData_fromRemoteAccompaniedCfg_isFirstChainedWrite(),
    .io_remoteXDMAData_fromRemoteAccompaniedCfg_isLastChainedWrite(),
    .io_remoteXDMAData_fromRemoteAccompaniedCfg_src(),
    .io_remoteXDMAData_fromRemoteAccompaniedCfg_dst(),
    .io_remoteXDMAData_toRemote_ready(1'b1),
    .io_remoteXDMAData_toRemote_valid(),
    .io_remoteXDMAData_toRemote_bits(),
    .io_remoteXDMAData_toRemoteAccompaniedCfg_readyToTransfer(),
    .io_remoteXDMAData_toRemoteAccompaniedCfg_taskID(),
    .io_remoteXDMAData_toRemoteAccompaniedCfg_length(),
    .io_remoteXDMAData_toRemoteAccompaniedCfg_taskType(),
    .io_remoteXDMAData_toRemoteAccompaniedCfg_isFirstChainedWrite(),
    .io_remoteXDMAData_toRemoteAccompaniedCfg_isLastChainedWrite(),
    .io_remoteXDMAData_toRemoteAccompaniedCfg_src(),
    .io_remoteXDMAData_toRemoteAccompaniedCfg_dst(),
    .io_remoteTaskFinished(1'b0),
    .io_status_readerBusy(),
    .io_status_writerBusy()
  );

endmodule
