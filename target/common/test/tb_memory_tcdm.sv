// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

/// Serve a bank of TCDM ports out of the simulation's main memory image.
///
/// The xDMA moves data between two xDMA endpoints, never between an engine and a
/// bus: a cluster reading main memory is really its xDMA talking to a second xDMA
/// that sits ON the memory and reaches it through TCDM ports. HeMAiA builds that
/// second endpoint into `hemaia_mem_system`, where real banked SRAM answers those
/// ports. This testbench has no such memory system -- main memory is the DPI image
/// that `fesvr` loads the ELF into -- so this module is what stands in for it,
/// presenting that same image on a TCDM port bank.
///
/// `BaseAddr` is added to every port address, because the endpoint's AGU emits
/// offsets into its own local memory while the DPI image is addressed globally.
///
/// WHAT THIS DOES NOT MODEL. Every port is granted every cycle. A real memory
/// system banks these ports and two ports meeting in one bank cost a cycle, so a
/// kernel measured against this sees a main memory with no bank conflicts and will
/// read slightly optimistic against silicon. That is deliberate: it keeps the far
/// side of the link out of the measurement, so what a run measures is the cluster.
/// Anything that concludes the far memory is the bottleneck has to be re-checked
/// against a banked model before it is believed.
module tb_memory_tcdm #(
  /// Number of TCDM ports served.
  parameter int unsigned NumPorts = 0,
  /// Data width of one port, in bits.
  parameter int unsigned DataWidth = 0,
  /// Added to every incoming port address to reach the global memory image.
  parameter longint unsigned BaseAddr = 0,
  /// Cycles between a granted request and its response, matching the local TCDM.
  parameter int unsigned ResponseLatency = 1,
  parameter type tcdm_req_t = logic,
  parameter type tcdm_rsp_t = logic
)(
  input  logic                     clk_i,
  input  logic                     rst_ni,
  input  tcdm_req_t [NumPorts-1:0] req_i,
  output tcdm_rsp_t [NumPorts-1:0] rsp_o
);

  import "DPI-C" function void tb_memory_read(
    input longint addr,
    input int len,
    output byte data[]
  );
  import "DPI-C" function void tb_memory_write(
    input longint addr,
    input int len,
    input byte data[],
    input bit strb[]
  );

  localparam int unsigned NumBytes = DataWidth / 8;
  localparam int unsigned BusAlign = $clog2(NumBytes);

  for (genvar i = 0; i < NumPorts; i++) begin : gen_port
    // Contention-free by construction, see the note above.
    assign rsp_o[i].q_ready = 1'b1;

    // Writes retire on the clock edge that accepts them.
    always_ff @(posedge clk_i) begin
      automatic byte data[NumBytes];
      automatic bit  strb[NumBytes];
      automatic longint unsigned addr;
      if (rst_ni && req_i[i].q_valid && req_i[i].q.write) begin
        addr = BaseAddr + longint'(req_i[i].q.addr);
        for (int b = 0; b < NumBytes; b++) begin
          // verilog_lint: waive-start always-ff-non-blocking
          data[b] = req_i[i].q.data[b*8+:8];
          strb[b] = req_i[i].q.strb[b];
          // verilog_lint: waive-stop always-ff-non-blocking
        end
        tb_memory_write((addr >> BusAlign) << BusAlign, NumBytes, data, strb);
      end
    end

    // Reads are sampled ON THE CLOCK EDGE, not combinationally, and the registers
    // below are what give the port the same request-to-data distance it would see
    // from the cluster's own TCDM.
    //
    // THE EDGE MATTERS FOR SIMULATION SPEED, not just for timing. A DPI call inside
    // an `always_comb` is re-evaluated whenever any input of the block settles, and
    // DPI is serialised against the rest of the model -- with one of these per port
    // it starves a threaded simulation of anything to parallelise and the whole run
    // collapses to the speed of the import calls. Clocked, each port makes at most
    // one call per cycle and only when it actually has a read outstanding.
    logic                 rvalid_q;
    logic [DataWidth-1:0] rdata_q;

    always_ff @(posedge clk_i or negedge rst_ni) begin
      automatic byte data[NumBytes];
      automatic longint unsigned addr;
      if (!rst_ni) begin
        rvalid_q <= 1'b0;
        rdata_q  <= '0;
      end else begin
        rvalid_q <= req_i[i].q_valid & ~req_i[i].q.write;
        if (req_i[i].q_valid && !req_i[i].q.write) begin
          addr = BaseAddr + longint'(req_i[i].q.addr);
          tb_memory_read((addr >> BusAlign) << BusAlign, NumBytes, data);
          for (int b = 0; b < NumBytes; b++) rdata_q[b*8+:8] <= data[b];
        end
      end
    end

    // One register is exactly `ResponseLatency` 1; deeper memories add stages here.
    if (ResponseLatency != 1) begin : gen_latency_check
      initial $fatal(1, "tb_memory_tcdm: only ResponseLatency 1 is modelled");
    end

    assign rsp_o[i].p_valid = rvalid_q;
    assign rsp_o[i].p.data  = rdata_q;
  end

  // The xDMA never issues atomics; if one appears the request has been mis-routed
  // and would be silently dropped rather than executed.
  `ifndef SYNTHESIS
  for (genvar i = 0; i < NumPorts; i++) begin : gen_no_amo_check
    always_ff @(posedge clk_i) begin
      if (rst_ni && req_i[i].q_valid && req_i[i].q.amo != reqrsp_pkg::AMONone) begin
        $fatal(1, "tb_memory_tcdm: atomic on port %0d; main memory serves no atomics", i);
      end
    end
  end
  `endif

endmodule
