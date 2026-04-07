
package snax.readerWriter

import chisel3._
import chisel3.util._

import snax.utils._

// Bundle for the writer-side write port into the reader's FixedLevelCache memory.
// Used only when the FixedLevelCache is instantiated in ReaderWriter mode (isReaderWriter = true).
class FixedLevelCacheWriterPort(fixedCacheDepth: Int, fixedCacheWidth: Int) extends Bundle {
  val enable = Input(Bool())
  val index  = Input(UInt(log2Ceil(fixedCacheDepth).W))
  val data   = Input(UInt(fixedCacheWidth.W))
}

/** FixedLevelCache with dual memory banks (even/odd) for concurrent read/write.
  *
  * The cache has two SyncReadMem banks:
  *   - Bank 0: stores data at even indices (index LSB = 0)
  *   - Bank 1: stores data at odd indices (index LSB = 1)
  *
  * Read instructions have priority over writes. When both target the same bank,
  * the read proceeds and the write stalls for one cycle.
  *
  * When useFixedCache is false, the cache is bypassed: dataInTCDM flows straight to dataOut.
  *
  * @param isReaderWriter When true, an additional writerPort is exposed for the ReaderWriter's
  *   writer to inject data directly into the cache.
  */
class FixedLevelCache(fixedCacheDepth: Int, fixedCacheWidth: Int, isReader: Boolean, isReaderWriter: Boolean = false) extends Module {
  val io = IO(new Bundle {
    // Standalone useCache signal (extracted from AGU, no longer per-instruction)
    val useFixedCache = Input(Bool())
    // Write instructions: write TCDM data into cache at the given index
    val writeFixedCacheInstruction = Flipped(Decoupled(new WriteFixedCacheInstructionIO(fixedCacheDepth)))
    // Read instructions: read cached data from the given index
    val readFixedCacheInstruction  = Flipped(Decoupled(new ReadFixedCacheInstructionIO(fixedCacheDepth)))
    // TCDM data input (paired with write instructions)
    val dataInTCDM = Flipped(Decoupled(UInt(fixedCacheWidth.W)))
    // When instantiated in ReaderWriter mode the writer can write directly into the cache memory
    val writerPort = if (isReaderWriter) Some(new FixedLevelCacheWriterPort(fixedCacheDepth, fixedCacheWidth)) else None
    // Data output (from cache reads or passthrough)
    val dataOut = Decoupled(UInt(fixedCacheWidth.W))
    val busy = Output(Bool())
  })

  dontTouch(io.dataOut)

  // ── Dual memory banks ──
  // Bank 0 holds even-indexed entries, Bank 1 holds odd-indexed entries
  val bankDepth = (fixedCacheDepth + 1) / 2  // ceil(fixedCacheDepth / 2)
  val mem0 = SyncReadMem(bankDepth, UInt(fixedCacheWidth.W))
  val mem1 = SyncReadMem(bankDepth, UInt(fixedCacheWidth.W))

  // When used inside a ReaderWriter, the writer can write directly into this cache via the writerPort.
  if (isReaderWriter) {
    when(io.writerPort.get.enable) {
      val wrBank = io.writerPort.get.index(0)
      val wrAddr = io.writerPort.get.index >> 1.U
      when(wrBank === 0.U) {
        mem0.write(wrAddr, io.writerPort.get.data)
      }.otherwise {
        mem1.write(wrAddr, io.writerPort.get.data)
      }
    }
  }

  // ── Bypass mode (no cache) ──
  when(!io.useFixedCache) {
    io.dataOut <> io.dataInTCDM
    io.writeFixedCacheInstruction.ready := true.B  // drain any stale instructions
    io.readFixedCacheInstruction.ready  := true.B
    io.busy := false.B
  }.otherwise {
    // ── Cache mode ──

    // Determine which bank each instruction targets
    val writeBank = io.writeFixedCacheInstruction.bits.index(0)
    val writeAddr = io.writeFixedCacheInstruction.bits.index >> 1.U

    // A write can proceed when: write + data valid AND target bank not busy with a read
    // (reads have priority over writes)
    val canAcceptWrite = Wire(Bool())

    // ── Local instruction register ──
    // Stores the current read instruction in a register, decoupling the instruction
    // buffer from the SyncReadMem read pipeline. The buffer is ready'd immediately
    // on acceptance, so the next instruction can be accepted and its SyncReadMem read
    // issued in the same cycle that the current data is delivered — no 1-cycle gap.
    // Only ONE read is issued per instruction (on the accept cycle). A hold register
    // keeps the data stable while waiting for downstream to consume it, freeing the
    // bank for writes on all other cycles.
    val instrValid = RegInit(false.B)
    val instrIndex = Reg(UInt(log2Ceil(fixedCacheDepth).W))

    // dataHeld: stays true once read data has arrived, until consumed (delivered)
    val dataHeld = RegInit(false.B)

    // Data arriving from SyncReadMem: one cycle after issueRead was asserted
    val readDataArriving = RegInit(false.B)

    val dataAvailable = instrValid && (dataHeld || readDataArriving)

    // Fire: data delivered to downstream
    val delivering = dataAvailable && io.dataOut.ready

    // Accept a new instruction when the register is free or being freed (delivering)
    val canAcceptNew = !instrValid || delivering
    val newBank = io.readFixedCacheInstruction.bits.index(0)
    val newAddr = io.readFixedCacheInstruction.bits.index >> 1.U
    val acceptNew = io.readFixedCacheInstruction.valid && canAcceptNew

    // Register update
    when(acceptNew) {
      instrValid := true.B
      instrIndex := io.readFixedCacheInstruction.bits.index
    }.elsewhen(delivering && !acceptNew) {
      instrValid := false.B
    }

    // ── Issue SyncReadMem read ──
    // Only when accepting a new instruction (one read per instruction).
    // No re-reads: the hold register keeps data stable while waiting for downstream.
    // This frees the bank for writes on all non-accept cycles.
    val issueRead   = acceptNew
    val readBankSel = newBank
    val readAddrSel = newAddr

    // Track data arrival (one cycle after read issued)
    readDataArriving := issueRead

    // dataHeld update (priority: acceptNew > readDataArriving > delivering)
    when(acceptNew) {
      dataHeld := false.B      // new instruction, data not ready yet
    }.elsewhen(readDataArriving) {
      dataHeld := true.B       // data arrived from memory, latch it
    }.elsewhen(delivering) {
      dataHeld := false.B      // data consumed
    }

    // Per-bank read busy flags (read priority: writes blocked when bank is reading)
    val bank0ReadBusy = issueRead && readBankSel === 0.U
    val bank1ReadBusy = issueRead && readBankSel === 1.U

    // A write can only proceed when its target bank is not busy with a read
    val writeBankBusy = Mux(writeBank === 0.U, bank0ReadBusy, bank1ReadBusy)
    canAcceptWrite := io.writeFixedCacheInstruction.valid && io.dataInTCDM.valid && !writeBankBusy

    // ── Bank 0: single readWrite port ──
    // mem.readWrite(addr, wdata, en, isWrite) provides a single port.
    // When en=true, isWrite=false: reads from addr, data available next cycle.
    // When en=true, isWrite=true: writes wdata to addr.
    // Read and write to the same bank are mutually exclusive (read has priority).
    val bank0WriteEn = canAcceptWrite && writeBank === 0.U
    val bank0En      = bank0ReadBusy || bank0WriteEn
    val bank0IsWrite = bank0WriteEn
    val bank0Addr    = Mux(bank0ReadBusy, readAddrSel, writeAddr)
    val bank0MemOut  = mem0.readWrite(bank0Addr, io.dataInTCDM.bits, bank0En, bank0IsWrite)

    // Hold register: captures read data on the cycle it arrives, holds it stable afterwards
    val bank0WasRead = RegNext(bank0ReadBusy, false.B)
    val bank0HoldReg = Reg(UInt(fixedCacheWidth.W))
    when(bank0WasRead) { bank0HoldReg := bank0MemOut }
    val bank0ReadData = Mux(bank0WasRead, bank0MemOut, bank0HoldReg)

    // ── Bank 1: single readWrite port ──
    val bank1WriteEn = canAcceptWrite && writeBank === 1.U
    val bank1En      = bank1ReadBusy || bank1WriteEn
    val bank1IsWrite = bank1WriteEn
    val bank1Addr    = Mux(bank1ReadBusy, readAddrSel, writeAddr)
    val bank1MemOut  = mem1.readWrite(bank1Addr, io.dataInTCDM.bits, bank1En, bank1IsWrite)

    val bank1WasRead = RegNext(bank1ReadBusy, false.B)
    val bank1HoldReg = Reg(UInt(fixedCacheWidth.W))
    when(bank1WasRead) { bank1HoldReg := bank1MemOut }
    val bank1ReadData = Mux(bank1WasRead, bank1MemOut, bank1HoldReg)

    // ── Read data output ──
    // Select output based on which bank was read on the PREVIOUS cycle (1-cycle SyncReadMem latency)
    val readBankReg = RegEnable(readBankSel, 0.U(1.W), issueRead)
    val readPipeData = Mux(readBankReg === 0.U, bank0ReadData, bank1ReadData)

    // ── Handshake signals ──
    io.readFixedCacheInstruction.ready := canAcceptNew
    io.writeFixedCacheInstruction.ready := canAcceptWrite
    io.dataInTCDM.ready := canAcceptWrite

    io.dataOut.valid := dataAvailable
    io.dataOut.bits  := readPipeData

    io.busy := instrValid || io.readFixedCacheInstruction.valid || io.writeFixedCacheInstruction.valid
  }
}
