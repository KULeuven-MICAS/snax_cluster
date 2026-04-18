
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
  * The cache has two single-port SRAM banks (via FixedCacheMemoryLib):
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
class FixedLevelCache(fixedCacheDepth: Int, fixedCacheWidth: Int, isReader: Boolean, isReaderWriter: Boolean = false, isSynthesis: Boolean = false) extends Module {
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
  val addrWidth = log2Ceil(bankDepth)

  // Bank control wires (active HIGH enables, driven by cache-mode logic)
  val bank0_writeEn   = WireDefault(false.B)
  val bank0_writeAddr = WireDefault(0.U(addrWidth.W))
  val bank0_writeData = WireDefault(0.U(fixedCacheWidth.W))
  val bank0_readEn    = WireDefault(false.B)
  val bank0_readAddr  = WireDefault(0.U(addrWidth.W))
  val bank1_writeEn   = WireDefault(false.B)
  val bank1_writeAddr = WireDefault(0.U(addrWidth.W))
  val bank1_writeData = WireDefault(0.U(fixedCacheWidth.W))
  val bank1_readEn    = WireDefault(false.B)
  val bank1_readAddr  = WireDefault(0.U(addrWidth.W))

  // Memory output data wires
  val mem0Q = Wire(UInt(fixedCacheWidth.W))
  val mem1Q = Wire(UInt(fixedCacheWidth.W))

  // WriterPort bank-busy flags (visible to cache-mode logic for stalling TCDM writes)
  val writerPortBank0Busy = WireDefault(false.B)
  val writerPortBank1Busy = WireDefault(false.B)

  if (isReaderWriter) {
    // ── Dual-port (1W1R) SRAMs: write port for writerPort + main writes, read port for reads ──
    val mem0 = Module(new FixedCacheDualPortMemoryLib(bankDepth, fixedCacheWidth, isSynthesis))
    val mem1 = Module(new FixedCacheDualPortMemoryLib(bankDepth, fixedCacheWidth, isSynthesis))

    // Write port defaults (disabled)
    mem0.io.web := true.B;  mem0.io.aa := 0.U; mem0.io.d := 0.U; mem0.io.bweb := 0.U
    mem1.io.web := true.B;  mem1.io.aa := 0.U; mem1.io.d := 0.U; mem1.io.bweb := 0.U
    // Read port defaults (disabled)
    mem0.io.reb := true.B;  mem0.io.ab := 0.U
    mem1.io.reb := true.B;  mem1.io.ab := 0.U

    // WriterPort drives the write port (higher priority via last-connect)
    // Main path writes are connected first (lower priority), writerPort overrides.
    // When writerPort is active on a bank, TCDM writes to that bank are blocked.
    when(io.writerPort.get.enable) {
      val wrBank = io.writerPort.get.index(0)
      val wrAddr = io.writerPort.get.index >> 1
      when(wrBank === 0.U) {
        writerPortBank0Busy := true.B
      }.otherwise {
        writerPortBank1Busy := true.B
      }
    }

    // Main path writes (lower priority, will be overridden by writerPort via last-connect)
    when(bank0_writeEn) {
      mem0.io.web := false.B; mem0.io.aa := bank0_writeAddr
      mem0.io.d := bank0_writeData; mem0.io.bweb := 0.U
    }
    when(bank1_writeEn) {
      mem1.io.web := false.B; mem1.io.aa := bank1_writeAddr
      mem1.io.d := bank1_writeData; mem1.io.bweb := 0.U
    }

    // WriterPort overrides main path writes (last-connect = highest priority)
    when(io.writerPort.get.enable) {
      val wrBank = io.writerPort.get.index(0)
      val wrAddr = io.writerPort.get.index >> 1
      when(wrBank === 0.U) {
        mem0.io.web := false.B; mem0.io.aa := wrAddr
        mem0.io.d := io.writerPort.get.data; mem0.io.bweb := 0.U
      }.otherwise {
        mem1.io.web := false.B; mem1.io.aa := wrAddr
        mem1.io.d := io.writerPort.get.data; mem1.io.bweb := 0.U
      }
    }

    // Read port (independent from write port — no conflicts)
    when(bank0_readEn) {
      mem0.io.reb := false.B; mem0.io.ab := bank0_readAddr
    }
    when(bank1_readEn) {
      mem1.io.reb := false.B; mem1.io.ab := bank1_readAddr
    }

    mem0Q := mem0.io.q
    mem1Q := mem1.io.q

  } else {
    // ── Single-port SRAMs ──
    val mem0 = Module(new FixedCacheMemoryLib(bankDepth, fixedCacheWidth, isSynthesis))
    val mem1 = Module(new FixedCacheMemoryLib(bankDepth, fixedCacheWidth, isSynthesis))

    // Defaults (disabled)
    mem0.io.ceb := true.B; mem0.io.web := true.B; mem0.io.a := 0.U; mem0.io.d := 0.U; mem0.io.bweb := 0.U
    mem1.io.ceb := true.B; mem1.io.web := true.B; mem1.io.a := 0.U; mem1.io.d := 0.U; mem1.io.bweb := 0.U

    // Writes (lower priority)
    when(bank0_writeEn) {
      mem0.io.ceb := false.B; mem0.io.web := false.B
      mem0.io.a := bank0_writeAddr; mem0.io.d := bank0_writeData; mem0.io.bweb := 0.U
    }
    when(bank1_writeEn) {
      mem1.io.ceb := false.B; mem1.io.web := false.B
      mem1.io.a := bank1_writeAddr; mem1.io.d := bank1_writeData; mem1.io.bweb := 0.U
    }

    // Reads (higher priority via last-connect, overrides writes on single port)
    when(bank0_readEn) {
      mem0.io.ceb := false.B; mem0.io.web := true.B; mem0.io.a := bank0_readAddr
    }
    when(bank1_readEn) {
      mem1.io.ceb := false.B; mem1.io.web := true.B; mem1.io.a := bank1_readAddr
    }

    mem0Q := mem0.io.q
    mem1Q := mem1.io.q
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
    val writeAddr = io.writeFixedCacheInstruction.bits.index >> 1

    // A write can proceed when: write + data valid AND target bank not busy with a read
    // (reads have priority over writes)
    val canAcceptWrite = Wire(Bool())

    // ── Local instruction register ──
    // Stores the current read instruction in a register, decoupling the instruction
    // buffer from the SRAM read pipeline. The buffer is ready'd immediately
    // on acceptance, so the next instruction can be accepted and its SRAM read
    // issued in the same cycle that the current data is delivered — no 1-cycle gap.
    // Only ONE read is issued per instruction (on the accept cycle). A hold register
    // keeps the data stable while waiting for downstream to consume it, freeing the
    // bank for writes on all other cycles.
    val instrValid = RegInit(false.B)
    val instrIndex = Reg(UInt(log2Ceil(fixedCacheDepth).W))

    // dataHeld: stays true once read data has arrived, until consumed (delivered)
    val dataHeld = RegInit(false.B)

    // Data arriving from SRAM: one cycle after issueRead was asserted
    val readDataArriving = RegInit(false.B)

    val dataAvailable = instrValid && (dataHeld || readDataArriving)

    // Fire: data delivered to downstream
    val delivering = dataAvailable && io.dataOut.ready

    // Accept a new instruction when the register is free or being freed (delivering)
    val canAcceptNew = !instrValid || delivering
    val newBank = io.readFixedCacheInstruction.bits.index(0)
    val newAddr = io.readFixedCacheInstruction.bits.index >> 1
    val acceptNew = io.readFixedCacheInstruction.valid && canAcceptNew

    // Register update
    when(acceptNew) {
      instrValid := true.B
      instrIndex := io.readFixedCacheInstruction.bits.index
    }.elsewhen(delivering && !acceptNew) {
      instrValid := false.B
    }

    // ── Issue SRAM read ──
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

    // For dual-port (isReaderWriter): no read/write port conflict, but writerPort
    // blocks TCDM writes to the same bank (they share the single write port)
    // For single-port: writes blocked when target bank is busy with a read
    val writeBankBusy = if (isReaderWriter) {
      Mux(writeBank === 0.U, writerPortBank0Busy, writerPortBank1Busy) || (issueRead && readBankSel === writeBank && readAddrSel === writeAddr)
    } else {
      Mux(writeBank === 0.U, bank0ReadBusy, bank1ReadBusy)
    }
    canAcceptWrite := io.writeFixedCacheInstruction.valid && io.dataInTCDM.valid && !writeBankBusy

    // ── Bank 0 ──
    val bank0WriteEn = canAcceptWrite && writeBank === 0.U
    when(bank0WriteEn) {
      bank0_writeEn   := true.B
      bank0_writeAddr := writeAddr
      bank0_writeData := io.dataInTCDM.bits
    }
    when(bank0ReadBusy) {
      bank0_readEn   := true.B
      bank0_readAddr := readAddrSel
    }
    val bank0MemOut = mem0Q

    // Hold register: captures read data on the cycle it arrives, holds it stable afterwards
    val bank0WasRead = RegNext(bank0ReadBusy, false.B)
    val bank0HoldReg = Reg(UInt(fixedCacheWidth.W))
    when(bank0WasRead) { bank0HoldReg := bank0MemOut }
    val bank0ReadData = Mux(bank0WasRead, bank0MemOut, bank0HoldReg)

    // ── Bank 1 ──
    val bank1WriteEn = canAcceptWrite && writeBank === 1.U
    when(bank1WriteEn) {
      bank1_writeEn   := true.B
      bank1_writeAddr := writeAddr
      bank1_writeData := io.dataInTCDM.bits
    }
    when(bank1ReadBusy) {
      bank1_readEn   := true.B
      bank1_readAddr := readAddrSel
    }
    val bank1MemOut = mem1Q

    val bank1WasRead = RegNext(bank1ReadBusy, false.B)
    val bank1HoldReg = Reg(UInt(fixedCacheWidth.W))
    when(bank1WasRead) { bank1HoldReg := bank1MemOut }
    val bank1ReadData = Mux(bank1WasRead, bank1MemOut, bank1HoldReg)

    // ── Read data output ──
    // Select output based on which bank was read on the PREVIOUS cycle (1-cycle SRAM latency)
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
