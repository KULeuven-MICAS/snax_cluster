package snax_acc.utils

import chisel3._
import chisel3.util._

class DecoupledCat2to1[T <: Data](aWidth: Int, bWidth: Int) extends Module{
      val io = IO(new Bundle {
    val in1 = Flipped(Decoupled(UInt(aWidth.W)))  // First decoupled input interface
    val in2 = Flipped(Decoupled(UInt(bWidth.W)))  // Second decoupled input interface
    val out = Decoupled(UInt((aWidth + bWidth).W)) // Decoupled output interface
  })

  // Combine the bits of in1 and in2, in1 in higher bits
  io.out.bits := Cat(io.in1.bits, io.in2.bits)

  // Output is valid only when both inputs are valid
  io.out.valid := io.in1.valid && io.in2.valid

  // Ready is asserted to inputs when the output is ready
  io.in1.ready := io.out.ready && io.out.valid
  io.in2.ready := io.out.ready && io.out.valid

}

class DecoupledSplit1to2(cWidth: Int, aWidth: Int, bWidth: Int) extends Module {
  require(cWidth == aWidth + bWidth, "cWidth must be the sum of aWidth and bWidth")

  val io = IO(new Bundle {
    val in = Flipped(Decoupled(UInt(cWidth.W))) // Large decoupled input (c)
    val out1 = Decoupled(UInt(aWidth.W))        // Smaller decoupled output (a)
    val out2 = Decoupled(UInt(bWidth.W))        // Smaller decoupled output (b)
  })

  // Split the input bits into two parts
  io.out1.bits := io.in.bits(cWidth - 1, bWidth) // Upper bits go to out1 (a)
  io.out2.bits := io.in.bits(bWidth - 1, 0)      // Lower bits go to out2 (b)

  // Both outputs are valid when the input is valid
  io.out1.valid := io.in.valid
  io.out2.valid := io.in.valid

  // Input is ready when both outputs are ready
  io.in.ready := io.out1.ready && io.out2.ready
}
