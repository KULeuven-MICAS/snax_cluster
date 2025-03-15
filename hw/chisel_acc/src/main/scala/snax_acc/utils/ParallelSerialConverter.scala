package snax_acc.utils

import chisel3._
import chisel3.util._

/** Parameter class for ParallelToSerial.
  *
  * @param parallelWidth
  *   The total width of the parallel input data.
  * @param serialWidth
  *   The width of each output serial chunk, must divide parallelWidth evenly.
  */
case class ParallelToSerialParams(
    parallelWidth: Int,
    serialWidth: Int
) {
  require(
    parallelWidth % serialWidth == 0,
    "parallelWidth must be an integer multiple of serialWidth."
  )
}

/** A module that sends a parallel input (via Decoupled I/O) out as multiple
  * serial chunks (also Decoupled I/O).
  */
class ParallelToSerial(val p: ParallelToSerialParams) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(UInt(p.parallelWidth.W)))
    val out = Decoupled(UInt(p.serialWidth.W))
  })

  // Calculate how many serial chunks form one parallel word.
  val factor = p.parallelWidth / p.serialWidth

  // Shift register to hold the parallel data while serializing.
  val shiftReg = RegInit(0.U(p.parallelWidth.W))
  // Counts how many chunks remain to be sent.
  val count = RegInit(0.U(log2Ceil(factor + 1).W))

  // Accept a new parallel word only if we have nothing left to send.
  io.in.ready := (count === 0.U)

  // Once we get a new parallel word, store it and prepare to send.
  when(io.in.fire) {
    shiftReg := io.in.bits
  }.elsewhen(io.out.fire) {
    // Shift the data to the right to get the next chunk.
    shiftReg := shiftReg >> p.serialWidth.U
  }

  // On handshake, shift to the next chunk and decrement count.
  when(io.in.fire) {
    count := factor.U
  }.elsewhen(io.out.fire) {
    count := count - 1.U
  }

  // The output is valid if there are chunks left to send.
  io.out.valid := (count > 0.U)
  // The current chunk to send is the least significant bits of shiftReg.
  io.out.bits := shiftReg(p.serialWidth - 1, 0)

}

/** Parameter class for SerialToParallel.
  *
  * @param serialWidth
  *   The width of each incoming serial data chunk.
  * @param parallelWidth
  *   The total width of the parallel output data. Must be a multiple of
  *   serialWidth.
  */
case class SerialToParallelParams(
    serialWidth: Int,
    parallelWidth: Int
) {
  require(
    parallelWidth % serialWidth == 0,
    "parallelWidth must be an integer multiple of serialWidth."
  )
}

/** A module that collects multiple serial inputs (via Decoupled I/O) and
  * outputs them as a single parallel word (also Decoupled I/O).
  *
  * This version defers output by one clock after receiving the final serial
  * chunk, ensuring that the shift register contains the correct concatenated
  * data.
  */
class SerialToParallel(val p: SerialToParallelParams) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(UInt(p.serialWidth.W)))
    val out = Decoupled(UInt(p.parallelWidth.W))
  })

  // Number of input chunks required to form one parallel output
  val factor: Int = p.parallelWidth / p.serialWidth

  // Registers to track incoming data and chunk count
  val shiftReg = RegInit(0.U(p.parallelWidth.W))
  val count = RegInit(0.U(log2Ceil(factor + 1).W))

  io.in.ready := count =/= factor.U

  when(io.in.fire && count =/= 0.U) {
    // Shift in the new serial bits at a position based on the count
    shiftReg := shiftReg | (io.in.bits << (count * p.serialWidth.U))
  }.elsewhen(io.in.fire) {
    // If we're at the first chunk, reset the shift register
    shiftReg := io.in.bits
  }

  when(io.out.fire) {
    count := 0.U
  }.elsewhen(io.in.fire) {
    count := count + 1.U
  }

  io.out.valid := count === factor.U

  io.out.bits := shiftReg
}
