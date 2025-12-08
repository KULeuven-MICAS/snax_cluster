
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
  
  // Register to hold the previous request that was accepted
  val storedRequest = RegInit(0.U.asTypeOf(new FixedCacheInstructionIO(fixedCacheDepth)))
  val storedDataIn = RegInit(0.U(fixedCacheWidth.W))
  
  // Wire to select between stored request or new input request
  val activeRequest = Wire(new FixedCacheInstructionIO(fixedCacheDepth))
  val activeDataIn = Wire(UInt(fixedCacheWidth.W))
  
  dontTouch(io.dataOut)
  
  when(!io.fixedLevelCacheRequest.bits.useCache) {
      // Mode 1: No cache - direct passthrough
      io.fixedLevelCacheRequest.ready := true.B
      io.dataOut <> io.dataInTCDM
      request_enable := false.B
      delayedValid := false.B
      delayedUpdateCache := false.B
      
      // Use new request directly
      activeRequest := io.fixedLevelCacheRequest.bits
      activeDataIn := io.dataInTCDM.bits
  }.elsewhen(io.fixedLevelCacheRequest.bits.updateCache) {
      // Mode 2: Update cache - write to memory and forward data with delay
      
      // Check if we can accept new request: dataOut is ready or not valid yet
      val canAcceptNew = !delayedValid || io.dataOut.ready
      
      // Select between stored request (when stalled) or new request (when can accept)
      when(canAcceptNew) {
        activeRequest := io.fixedLevelCacheRequest.bits
        activeDataIn := io.dataInTCDM.bits
      }.otherwise {
        activeRequest := storedRequest
        activeDataIn := storedDataIn
      }
      
      // Keep request_enable high as long as inputs are valid, even when stalled
      request_enable := Mux(canAcceptNew, 
        io.fixedLevelCacheRequest.valid && io.dataInTCDM.valid,
        true.B  // Keep enabled when using stored request
      )
      
      io.fixedLevelCacheRequest.ready := io.fixedLevelCacheRequest.valid && io.dataInTCDM.valid && canAcceptNew
      io.dataInTCDM.ready := io.fixedLevelCacheRequest.valid && io.dataInTCDM.valid && canAcceptNew
      
      // Store request when accepting new one
      when(canAcceptNew && io.fixedLevelCacheRequest.valid && io.dataInTCDM.valid) {
        storedRequest := io.fixedLevelCacheRequest.bits
        storedDataIn := io.dataInTCDM.bits
      }
      
      // Delay the data, valid, and mode by one cycle to match memory read latency
      when(canAcceptNew && io.fixedLevelCacheRequest.valid && io.dataInTCDM.valid) {
        delayedValid := true.B
        delayedDataBits := io.dataInTCDM.bits
        delayedUpdateCache := true.B
      }.elsewhen(delayedValid && io.dataOut.ready) {
        delayedValid := false.B
      }
      
      io.dataOut.valid := delayedValid
      // Output selection based on delayed mode
      io.dataOut.bits := Mux(delayedUpdateCache, delayedDataBits, output_bits)
  }.otherwise{
      // Mode 3: Read from cache - output comes from memory with one-cycle latency
      
      // Check if we can accept new request: dataOut is ready or not valid yet
      val canAcceptNew = !delayedValid || io.dataOut.ready
      
      // Select between stored request (when stalled) or new request (when can accept)
      when(canAcceptNew) {
        activeRequest := io.fixedLevelCacheRequest.bits
        activeDataIn := io.dataInTCDM.bits
      }.otherwise {
        activeRequest := storedRequest
        activeDataIn := storedDataIn
      }
      
      // Keep request_enable high as long as input is valid, even when stalled
      request_enable := Mux(canAcceptNew,
        io.fixedLevelCacheRequest.valid,
        true.B  // Keep enabled when using stored request
      )
      
      io.fixedLevelCacheRequest.ready := io.fixedLevelCacheRequest.valid && canAcceptNew
      io.dataInTCDM.ready := false.B
      
      // Store request when accepting new one
      when(canAcceptNew && io.fixedLevelCacheRequest.valid) {
        storedRequest := io.fixedLevelCacheRequest.bits
      }
      
      // Only update when accepting new request
      when(canAcceptNew && io.fixedLevelCacheRequest.valid) {
        delayedValid := true.B
        delayedUpdateCache := false.B
      }.elsewhen(delayedValid && io.dataOut.ready) {
        delayedValid := false.B
      }
      
      io.dataOut.valid := delayedValid
      // Output selection based on delayed mode
      io.dataOut.bits := Mux(delayedUpdateCache, delayedDataBits, output_bits)
  }
  
  val mem = SyncReadMem(fixedCacheDepth, UInt(fixedCacheWidth.W))
  if (!isReader) {
      mem.write(io.fixedLevelCacheRequest.bits.index - 1.U, io.dataInAccelerator.get) //TODO: minus one is an oversimplification, probably an entire seperate address generator is needed
  }
  output_bits := mem.readWrite(activeRequest.index, activeDataIn, request_enable, activeRequest.updateCache)

  io.busy := (io.fixedLevelCacheRequest.valid || delayedValid) && io.fixedLevelCacheRequest.bits.useCache
}
