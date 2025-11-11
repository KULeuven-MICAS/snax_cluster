package snax.readerWriter

import chisel3._
import chisel3.util._

import snax.utils._

class FixedLevelCache(fixedCacheDepth: Int, fixedCacheWidth: Int, isReader: Boolean) extends Module {
  val io = IO(new Bundle {
    val fixedLevelCacheRequest = Flipped(Decoupled(new FixedCacheInstructionIO(fixedCacheDepth)))
    val dataInTCDM = Flipped(Decoupled(UInt(fixedCacheWidth.W)))
    val dataInAccelerator = if (isReader) None else Some(Input(UInt(fixedCacheWidth.W)))
    val dataOut = Decoupled(UInt(fixedCacheWidth.W))
    val busy = Output(Bool())
  })
    // Instantiates the fixed level cache memory.
    // For a reader, only one port is needed. Reading and writing cannot happen at the same time.
    // For a writer, two ports are needed. One for writing from the accelerator, one for reading to the accelerator.

  val request_enable = Wire(Bool())
  val output_bits = Wire(UInt(fixedCacheWidth.W))
  
  // Delay valid signal and mode by one cycle for cache operations to match SyncReadMem latency
  val delayedValid = RegInit(false.B)
  val delayedDataBits = RegInit(0.U(fixedCacheWidth.W))
  val delayedUpdateCache = RegInit(false.B)
  
  dontTouch(io.dataOut)
  
  when(!io.fixedLevelCacheRequest.bits.useCache) {
      // Mode 1: No cache - direct passthrough
      io.fixedLevelCacheRequest.ready := true.B
      io.dataOut <> io.dataInTCDM
      request_enable := false.B
      delayedValid := false.B
      delayedUpdateCache := false.B
  }.elsewhen(io.fixedLevelCacheRequest.bits.updateCache) {
      // Mode 2: Update cache - write to memory and forward data with delay
      request_enable := io.fixedLevelCacheRequest.valid && io.dataInTCDM.valid
      io.fixedLevelCacheRequest.ready := request_enable
      io.dataInTCDM.ready := request_enable
      
      // Delay the data, valid, and mode by one cycle to match memory read latency
      delayedValid := request_enable
      delayedDataBits := io.dataInTCDM.bits
      delayedUpdateCache := true.B
      
      io.dataOut.valid := delayedValid
      // Output selection based on delayed mode
      io.dataOut.bits := Mux(delayedUpdateCache, delayedDataBits, output_bits)
  }.otherwise{
      // Mode 3: Read from cache - output comes from memory with one-cycle latency
      request_enable := io.fixedLevelCacheRequest.valid
      io.fixedLevelCacheRequest.ready := request_enable
      io.dataInTCDM.ready := false.B
      
      delayedValid := request_enable
      delayedUpdateCache := false.B
      
      io.dataOut.valid := delayedValid
      // Output selection based on delayed mode
      io.dataOut.bits := Mux(delayedUpdateCache, delayedDataBits, output_bits)
  }
  
  val mem = SyncReadMem(fixedCacheDepth, UInt(fixedCacheWidth.W))
  if (!isReader) {
      mem.write(io.fixedLevelCacheRequest.bits.index - 1.U, io.dataInAccelerator.get) //TODO: minus one is an oversimplification, probably an entire seperate address generator is needed
  }
  output_bits := mem.readWrite(io.fixedLevelCacheRequest.bits.index, io.dataInTCDM.bits, request_enable, io.fixedLevelCacheRequest.bits.updateCache)

  io.busy := (io.fixedLevelCacheRequest.valid || delayedValid) && io.fixedLevelCacheRequest.bits.useCache
}