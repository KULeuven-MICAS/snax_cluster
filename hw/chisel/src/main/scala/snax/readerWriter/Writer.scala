package snax.readerWriter

import chisel3._
import chisel3.util._

import snax.utils._

class Writer(param: ReaderWriterParam, isReaderWriter: Boolean, moduleNamePrefix: String = "unnamed_cluster")
    extends Module
    with RequireAsyncReset {

  override val desiredName = s"${moduleNamePrefix}_Writer"

  val io = IO(new WriterIO(param, isReaderWriter))

  // New Address Generator
  val addressgen = Module(
    new AddressGenUnit(
      param.aguParam,
      moduleNamePrefix = s"${moduleNamePrefix}_Writer"
    )
  )

  // Write Requestors
  // Requestors to send address and data to TCDM
  val requestors = Module(
    new DataRequestors(
      tcdmDataWidth    = param.tcdmParam.dataWidth,
      tcdmAddressWidth = param.tcdmParam.addrWidth,
      numChannel       = param.tcdmParam.numChannel,
      isReader         = false,
      moduleNamePrefix = s"${moduleNamePrefix}_Writer"
    )
  )

  val dataBuffer = Module(
    new ComplexQueueConcat(
      inputWidth  = param.tcdmParam.dataWidth * param.tcdmParam.numChannel,
      outputWidth = param.tcdmParam.dataWidth,
      depth       = param.bufferDepth,
      pipe        = false
    ) {
      override val desiredName = s"${moduleNamePrefix}_Writer_DataBuffer"
    }
  )

  addressgen.io.cfg   := io.aguCfg
  addressgen.io.start := io.start

  if (!isReaderWriter) {
    // Standalone Writer: expose fixedCacheInstruction from AGU on the IO port so it can be
    // connected externally (e.g. to a reader's fixedCacheInstruction input).
    io.fixedCacheInstruction <> addressgen.io.fixedCacheInstruction
  } else {
    // In ReaderWriter mode, fixedCacheInstruction is consumed internally by the routing logic below.
    // The exposed IO port is tied off (unused by external logic).
    io.fixedCacheInstruction.valid := false.B
    io.fixedCacheInstruction.bits  := 0.U.asTypeOf(io.fixedCacheInstruction.bits)
  }
  // In ReaderWriter mode, fixedCacheInstruction.ready (AGU side) is driven by the data routing logic below

  // addrgen <> requestors
  requestors.io.zip(addressgen.io.addr).foreach {
    case (requestor, addrgen) => {
      requestor.in.addr <> addrgen
    }
  }

  // enabledChannel & enabledByteMask
  if (param.configurableChannel)
    requestors.io.zip(io.readerwriterCfg.enabledChannel.asBools).foreach {
      case (requestor, enable) => {
        requestor.enable := enable
      }
    }
  else requestors.io.foreach(_.enable := true.B)

  if (param.configurableByteMask)
    requestors.io.foreach(_.in.strb := io.readerwriterCfg.enabledByte)
  else
    requestors.io.zipWithIndex.foreach {
      case (requestor, _) => {
        requestor.in.strb := Fill(requestor.in.strb.getWidth, 1.U)
      }
    }

  // Requestor <> TCDM
  requestors.io.zip(io.tcdmReq).foreach {
    case (requestor, tcdmReq) => {
      requestor.out.tcdmReq <> tcdmReq
    }
  }

  // Requestor <> DataBuffer, Data Link
  requestors.io.zip(dataBuffer.io.out).foreach {
    case (requestor, dataBuffer) => {
      requestor.in.data.get <> dataBuffer
    }
  }

  // DataBuffer <> Input (with optional cache routing in ReaderWriter mode)
  if (!isReaderWriter) {
    // Standalone Writer: data always goes to the dataBuffer → TCDM
    if (param.crossClockDomain == false) {
      dataBuffer.io.in.head <> io.data
    } else {
      val clockDomainCrosser = Module(
        new AsyncQueue(chiselTypeOf(dataBuffer.io.in.head.bits)) {
          override val desiredName =
            s"${moduleNamePrefix}_Writer_ClockDomainCrosser"
        }
      )
      clockDomainCrosser.io.enq.clock := io.accClock.get
      clockDomainCrosser.io.deq.clock := clock
      clockDomainCrosser.io.enq.data <> io.data
      dataBuffer.io.in.head <> clockDomainCrosser.io.deq.data
    }
  } else {
    // ReaderWriter mode: route data to reader's FixedLevelCache write port or dataBuffer
    // based on the fixedCacheInstruction from the AGU.
    //
    // Routing rules (when useCache=true, i.e. fixed-cache mode is active):
    //   - lastAccess=false  (not the last time this data is accessed):
    //       Write data into the reader's FixedLevelCache at the given index.
    //       Do NOT forward to TCDM dataBuffer.
    //   - lastAccess=true   (last access – zero-stride counters at their ceil-1):
    //       Forward data to TCDM dataBuffer.
    //       Also write to the reader's cache so the reader can serve all iterations.
    // When useCache=false: bypass cache, always go to dataBuffer (normal path).

    // Intermediate decoupled wire that holds data after optional clock crossing.
    // We only connect valid and bits here; ready is driven by the routing logic below.
    val dataAfterCrosser = Wire(Decoupled(UInt((param.tcdmParam.dataWidth * param.tcdmParam.numChannel).W)))

    if (param.crossClockDomain == false) {
      dataAfterCrosser.valid := io.data.valid
      dataAfterCrosser.bits  := io.data.bits
      io.data.ready          := dataAfterCrosser.ready
    } else {
      val clockDomainCrosser = Module(
        new AsyncQueue(chiselTypeOf(dataBuffer.io.in.head.bits)) {
          override val desiredName =
            s"${moduleNamePrefix}_Writer_ClockDomainCrosser"
        }
      )
      clockDomainCrosser.io.enq.clock := io.accClock.get
      clockDomainCrosser.io.deq.clock := clock
      clockDomainCrosser.io.enq.data <> io.data
      dataAfterCrosser.valid                 := clockDomainCrosser.io.deq.data.valid
      dataAfterCrosser.bits                  := clockDomainCrosser.io.deq.data.bits
      clockDomainCrosser.io.deq.data.ready   := dataAfterCrosser.ready
    }

    // Both data and fixedCacheInstruction must be valid together before we can route.
    // However, when fixed-cache is disabled at runtime the AGU never emits instructions;
    // detect this and fall back to the direct TCDM path so data still flows.
    val instrValid = addressgen.io.fixedCacheInstruction.valid
    val instr      = addressgen.io.fixedCacheInstruction.bits

    // Default (no-cache path) — overridden below when fixedCache is active.
    io.fixedCacheWriterPort.get.enable            := false.B
    io.fixedCacheWriterPort.get.index             := 0.U
    io.fixedCacheWriterPort.get.data              := dataAfterCrosser.bits
    dataBuffer.io.in.head.valid                   := false.B
    dataBuffer.io.in.head.bits                    := dataAfterCrosser.bits
    dataAfterCrosser.ready                        := false.B
    addressgen.io.fixedCacheInstruction.ready     := false.B

    when(!io.aguCfg.enableFixedCache) {
      // ── No fixed-cache mode ──────────────────────────────────────────────────
      // Data streams straight to the TCDM dataBuffer, exactly as a standalone writer.
      // The fixedCacheInstruction FIFO is empty by design; consume it opportunistically
      // to avoid any backpressure side-effects.
      dataBuffer.io.in.head.valid               := dataAfterCrosser.valid
      dataAfterCrosser.ready                    := dataBuffer.io.in.head.ready
      addressgen.io.fixedCacheInstruction.ready := true.B
    }.otherwise {
      // ── Fixed-cache mode ─────────────────────────────────────────────────────
      // Route based on the instruction produced by the AGU alongside each address.
      //
      //  useCache=true,  lastAccess=false  → cache write only (skip TCDM for this iter)
      //  useCache=true,  lastAccess=true   → cache write + TCDM dataBuffer (last iter)
      //  useCache=false                    → TCDM dataBuffer only (normal)
      val goToTCDM  = !instr.useCache || instr.lastAccess
      val goToCache = instr.useCache
      val bothValid = dataAfterCrosser.valid && instrValid

      io.fixedCacheWriterPort.get.enable        := bothValid && goToCache
      io.fixedCacheWriterPort.get.index         := instr.index
      io.fixedCacheWriterPort.get.data          := dataAfterCrosser.bits
      dataBuffer.io.in.head.valid               := bothValid && goToTCDM

      // Cache writes are combinational (no backpressure); TCDM writes need buffer space.
      val targetReady = Mux(goToTCDM, dataBuffer.io.in.head.ready, true.B)
      dataAfterCrosser.ready                    := instrValid && targetReady
      addressgen.io.fixedCacheInstruction.ready := dataAfterCrosser.valid && targetReady
    }
  } // end ReaderWriter mode
  // Busy Signal
  io.busy := addressgen.io.busy | (~addressgen.io.bufferEmpty)

  // The debug signal from the dataBuffer to see if AGU and requestor work correctly: It should be high when valid signal at the combined output is low
  io.bufferEmpty := addressgen.io.bufferEmpty & dataBuffer.io.allEmpty
}

object WriterPrinter extends App {
  println(getVerilogString(new Writer(new ReaderWriterParam, false)))
}

object WriterEmitter extends App {
  emitVerilog(
    new Writer(new ReaderWriterParam, false),
    Array("--target-dir", "generated")
  )
}
