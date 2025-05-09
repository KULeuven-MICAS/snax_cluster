package snax_acc.spatial_array

import chisel3._
import chisel3.util._

class DecoupledCatNto1(
  dataWidths: Seq[Int]
) extends Module {
  require(dataWidths.nonEmpty, "Must have at least one input.")

  val totalWidth = dataWidths.sum

  val io = IO(new Bundle {
    val in = MixedVec((0 until dataWidths.length).map { i =>
      Flipped(Decoupled(UInt(dataWidths(i).W)))
    }) // Each entry should have its specific width
    val out = Decoupled(UInt(totalWidth.W))
  })

  // Concatenating input bits to form output UInt
  io.out.bits := io.in.map(_.bits).reduce(Cat(_, _))

  // Output is valid only when all inputs are valid
  io.out.valid := io.in.map(_.valid).reduce(_ && _)

  // Ready is asserted to inputs when the output is ready
  io.in.foreach(_.ready := io.out.ready && io.out.valid)
}
