
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
  * Read instructions (from readFixedCacheInstruction) have priority.
  * When a read targets a bank, that bank is busy and cannot be used for writes.
  * Write instructions (from writeFixedCacheInstruction + dataInTCDM) can only proceed
  * when their target bank is not busy being read.
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

    // Write pipeline: tracks that a write was issued last cycle
    val writePipeValid = RegInit(false.B)

    // Determine which bank each instruction targets
    val readBank  = io.readFixedCacheInstruction.bits.index(0)
    val readAddr  = io.readFixedCacheInstruction.bits.index >> 1.U
    val writeBank = io.writeFixedCacheInstruction.bits.index(0)
    val writeAddr = io.writeFixedCacheInstruction.bits.index >> 1.U

    // ── Read logic ──
    // The read is issued every cycle while the instruction is valid.
    // SyncReadMem has 1-cycle latency: data appears one cycle after the read is issued.
    // The read keeps being re-issued (same address) until downstream accepts the data.
    val canIssueRead = io.readFixedCacheInstruction.valid

    // A write can proceed when: write + data valid AND target bank not busy with a read AND write pipe empty
    val writeBankBusy = canIssueRead && (writeBank === readBank)
    val canAcceptWrite = io.writeFixedCacheInstruction.valid && io.dataInTCDM.valid && !writeBankBusy && !writePipeValid

    // ── Bank 0 operations ──
    val bank0ReadEn  = canIssueRead && readBank === 0.U
    val bank0WriteEn = canAcceptWrite && writeBank === 0.U
    val bank0ReadData = Wire(UInt(fixedCacheWidth.W))

    when(bank0ReadEn) {
      bank0ReadData := mem0.read(readAddr, true.B)
    }.elsewhen(bank0WriteEn) {
      mem0.write(writeAddr, io.dataInTCDM.bits)
      bank0ReadData := DontCare
    }.otherwise {
      bank0ReadData := DontCare
    }

    // ── Bank 1 operations ──
    val bank1ReadEn  = canIssueRead && readBank === 1.U
    val bank1WriteEn = canAcceptWrite && writeBank === 1.U
    val bank1ReadData = Wire(UInt(fixedCacheWidth.W))

    when(bank1ReadEn) {
      bank1ReadData := mem1.read(readAddr, true.B)
    }.elsewhen(bank1WriteEn) {
      mem1.write(writeAddr, io.dataInTCDM.bits)
      bank1ReadData := DontCare
    }.otherwise {
      bank1ReadData := DontCare
    }

    // ── Read data output ──
    // SyncReadMem data appears one cycle after the read is issued.
    // readDataValid goes high one cycle after canIssueRead, and goes low for one cycle
    // after an instruction is consumed (fire) so the SyncReadMem can fetch the next address.
    val readBankReg = RegEnable(readBank, 0.U(1.W), canIssueRead)
    val readPipeData = Mux(readBankReg === 0.U, bank0ReadData, bank1ReadData)

    val readDataValid = RegNext(canIssueRead && !io.readFixedCacheInstruction.fire, false.B)

    // ── Write pipeline ──
    when(canAcceptWrite) {
      writePipeValid := true.B
    }.otherwise {
      writePipeValid := false.B
    }

    // ── Handshake signals ──
    // dataOut.valid is asserted independently of dataOut.ready (proper Decoupled).
    // The read instruction is consumed only when downstream accepts the data.
    io.readFixedCacheInstruction.ready  := readDataValid && io.dataOut.ready
    io.writeFixedCacheInstruction.ready := canAcceptWrite
    io.dataInTCDM.ready                := canAcceptWrite

    // Output from read
    io.dataOut.valid := readDataValid
    io.dataOut.bits  := readPipeData

    io.busy := io.readFixedCacheInstruction.valid || readDataValid || io.writeFixedCacheInstruction.valid || writePipeValid
  }
}
