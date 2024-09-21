package snax.utils

import chisel3._
import chisel3.util._
import chisel3.reflect.DataMirror

/** The definition of -|> / -||> / -|||> connector for decoupled signal it
  * connects leftward Decoupled signal (Decoupled port) and rightward Decoupled
  * signal (Flipped port); and insert one level of pipeline in between to avoid
  * long combinatorial datapath
  */

class DataCut[T <: Data](gen: T, delay: Int, pipeline: Boolean = false)
    extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(gen))
    val out = Decoupled(gen)
  })
  val cuts = Seq.fill(delay)(Module(new Queue(gen, 1, pipe = true)))

  io.in <> cuts.head.io.enq
  cuts.zip(cuts.tail).foreach { case (left, right) =>
    left.io.deq <> right.io.enq
  }
  cuts.last.io.deq <> io.out

}

object DecoupledCut {
  implicit class BufferedDecoupledConnectionOp[T <: Data](
      val left: DecoupledIO[T]
  ) {
    // This class defines the implicit class for the new operand -|>,-||>, -|||> for DecoupleIO

    def -|>(
        right: DecoupledIO[T]
    )(implicit sourceInfo: chisel3.experimental.SourceInfo): Unit = {
      val buffer = Module(
        new Queue(chiselTypeOf(left.bits), entries = 1, pipe = false)
      )
      buffer.suggestName("fullCut1")

      left <> buffer.io.enq
      buffer.io.deq <> right
    }

    def -||>(
        right: DecoupledIO[T]
    )(implicit sourceInfo: chisel3.experimental.SourceInfo): Unit = {
      val buffer = Module(
        new Queue(chiselTypeOf(left.bits), entries = 2, pipe = false)
      )
      buffer.suggestName("fullCut2")
      left <> buffer.io.enq
      buffer.io.deq <> right
    }

    def -|||>(
        right: DecoupledIO[T]
    )(implicit sourceInfo: chisel3.experimental.SourceInfo): Unit = {
      val buffer = Module(
        new Queue(chiselTypeOf(left.bits), entries = 3, pipe = false)
      )
      buffer.suggestName("fullCut3")
      left <> buffer.io.enq
      buffer.io.deq <> right
    }

    def -\>(
        right: DecoupledIO[T]
    )(implicit sourceInfo: chisel3.experimental.SourceInfo): Unit = {
      val buffer = Module(
        new DataCut(chiselTypeOf(left.bits), delay = 1, pipeline = true)
      )
      buffer.suggestName("dataCut1")

      left <> buffer.io.in
      buffer.io.out <> right
    }

    def -\\>(
        right: DecoupledIO[T]
    )(implicit sourceInfo: chisel3.experimental.SourceInfo): Unit = {
      val buffer = Module(
        new DataCut(chiselTypeOf(left.bits), delay = 2, pipeline = true)
      )
      buffer.suggestName("dataCut2")

      left <> buffer.io.in
      buffer.io.out <> right
    }

    def -\\\>(
        right: DecoupledIO[T]
    )(implicit sourceInfo: chisel3.experimental.SourceInfo): Unit = {
      val buffer = Module(
        new DataCut(chiselTypeOf(left.bits), delay = 3, pipeline = true)
      )
      buffer.suggestName("dataCut3")

      left <> buffer.io.in
      buffer.io.out <> right
    }
  }
}

object BitsConcat {
  implicit class UIntConcatOp[T <: Bits](val left: T) {
    // This class defines the implicit class for the new operand ++ for UInt
    def ++(
        right: T
    )(implicit sourceInfo: chisel3.experimental.SourceInfo): T =
      Cat(left, right).asInstanceOf[T]
  }
}
