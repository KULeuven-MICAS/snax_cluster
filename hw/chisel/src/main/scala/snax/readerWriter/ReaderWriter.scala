package snax.readerWriter

import chisel3._

import snax.utils._

// ReaderWriter is the module that has a reader port and writer port, but they share one TCDM interface.
// This is suitable for the case that the throughput is not high.

class ReaderWriter(
  readerParam:      ReaderWriterParam,
  writerParam:      ReaderWriterParam,
  moduleNamePrefix: String = "unnamed_cluster"
) extends Module
    with RequireAsyncReset {

  override val desiredName = s"${moduleNamePrefix}_ReaderWriter"

  val io = IO(new ReaderWriterIO(readerParam, writerParam))

  // Reader
  val reader = Module(
    new Reader(
      readerParam,
      moduleNamePrefix = s"${moduleNamePrefix}_RWReader"
    )
  )

  reader.io.aguCfg               := io.readerInterface.aguCfg
  reader.io.readerwriterCfg      := io.readerInterface.readerwriterCfg
  reader.io.data <> io.readerInterface.data
  reader.io.start                := io.readerInterface.start
  io.readerInterface.busy        := reader.io.busy
  io.readerInterface.bufferEmpty := reader.io.bufferEmpty

  // Writer
  val writer = Module(
    new Writer(
      writerParam,
      moduleNamePrefix = s"${moduleNamePrefix}_RWWriter"
    )
  )

  writer.io.aguCfg               := io.writerInterface.aguCfg
  writer.io.readerwriterCfg      := io.writerInterface.readerwriterCfg
  writer.io.data <> io.writerInterface.data
  writer.io.start                := io.writerInterface.start
  io.writerInterface.busy        := writer.io.busy
  io.writerInterface.bufferEmpty := writer.io.bufferEmpty

  // Both reader and writer share the same Request interface
  val readerwriterMux = Seq.fill(readerParam.tcdmParam.numChannel)(
    Module(
      new MuxDecoupled(
        new SparseTCDMReq(
          readerParam.tcdmParam.addrWidth,
          readerParam.tcdmParam.dataWidth
        ),
        2
      )
    )
  )

  // Writer is put at 0th input
  readerwriterMux.zip(writer.io.tcdmReq).foreach { case (mux, writerReq) =>
    mux.io.in(0) <> writerReq
  }

  // Reader is put at 1st input
  readerwriterMux.zip(reader.io.tcdmReq).foreach { case (mux, readerReq) =>
    mux.io.in(1) <> readerReq
  }

  // Connect the DecoupledMux to the TCDM interface
  readerwriterMux.zip(io.readerInterface.tcdmReq).foreach { case (mux, tcdmReq) =>
    tcdmReq <> mux.io.out
  }

  // Channel Selection Logic
  // The writer wins whenever it wants the port; the reader only gets it in the gaps.
  val selComb = Mux(writer.io.tcdmReq.map(_.valid).reduce(_ || _), 0.U, 1.U)

  // ...but not *mid-request*. `sel` steers both the payload and the ready of every mux, so
  // letting it follow `selComb` combinationally swaps the request under a held valid: a reader
  // request that is asserted and still waiting for grant becomes a writer request the moment the
  // writer raises valid. The TCDM `stream_xbar` checks
  // `valid_i && !ready_o |=> $stable(data_i)` (and the same for its own `sel_i`), so that swap
  // trips `input_data_unstable`. It survives here only because this interconnect arbitrates
  // combinationally every cycle and simply re-arbitrates the new payload -- nothing latched the
  // old one -- so the stalled request is retried rather than lost. An interconnect that samples
  // the request when it first sees valid would mis-route it. `DataRequestor` holds the priority
  // hint for this same reason; `sel` needs the same treatment.
  //
  // Hold across a stall, but switch in the same cycle when nothing is in flight, so the common
  // case costs no bubble. Both terms are registered, so `sel` is not combinational on `ready`.
  //
  // Per CHANNEL, not one shared hold: each mux has its own `ready` from the TCDM arbiter, and the
  // assertion is per xbar input, so only the channel that actually stalled needs pinning. Holding
  // all 16 because one of them lost arbitration would keep the writer off the whole port for as
  // long as any single reader request waits, which measurably costs FlashAttention. The channels
  // are independent downstream too -- separate address queues, separate response credits, and the
  // response gate below is already a per-channel register -- so they may disagree on `sel` for a
  // cycle without anything being reassembled out of order.
  val selHold    = RegInit(VecInit(Seq.fill(readerParam.tcdmParam.numChannel)(1.U(1.W))))
  val reqStalled = RegInit(VecInit(Seq.fill(readerParam.tcdmParam.numChannel)(false.B)))
  val sel        = (0 until readerParam.tcdmParam.numChannel).map { i =>
    Mux(reqStalled(i), selHold(i), selComb)
  }
  readerwriterMux.zip(sel).foreach { case (mux, s) => mux.io.sel := s }

  // In flight exactly when the assertion's antecedent holds on this channel. A channel that fired
  // is free to present its next address, because its ready was high on the cycle it changed.
  readerwriterMux.zip(sel).zipWithIndex.foreach { case ((mux, s), i) =>
    reqStalled(i) := mux.io.out.valid && !mux.io.out.ready
    selHold(i)    := s
  }

  // Connect the response from TCDM to the reader
  reader.io.tcdmRsp.zip(io.readerInterface.tcdmRsp).zip(sel).foreach {
    case ((reader, interface), s) => {
      // Bits is connected directly
      reader.bits  := interface.bits
      // Valid is connected with the interface valid, under the condition that the last request on
      // THIS channel was the reader's -- each channel chooses for itself, so the gate follows it.
      reader.valid := interface.valid && RegNext(s === 1.U)
    }
  }
}

object ReaderWriterEmitter extends App {
  println(
    getVerilogString(
      new ReaderWriter(new ReaderWriterParam, new ReaderWriterParam)
    )
  )
}
