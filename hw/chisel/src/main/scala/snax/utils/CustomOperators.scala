package snax.utils

import chisel3._
import chisel3.util._
import chisel3.reflect.DataMirror

/** The definition of -|> / -||> / -|||> connector for decoupled signal it
  * connects leftward Decoupled signal (Decoupled port) and rightward Decoupled
  * signal (Flipped port); and insert one level of pipeline in between to avoid
  * long combinatorial datapath
  */

class DecoupledCut[T <: Data](gen: T, delay: Int, pipeline: Boolean = false)
    extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(gen))
    val out = Decoupled(gen)
  })
  val headCut =
    if (pipeline) Module(new Queue(gen, 1, pipe = true))
    else Module(new Queue(gen, 2, pipe = false))
  val tailCuts = Seq.fill(delay - 1)(Module(new Queue(gen, 1, pipe = true)))

  io.in <> headCut.io.enq
  if (tailCuts.isEmpty) headCut.io.deq <> io.out
  else {
    headCut.io.deq <> tailCuts.head.io.enq
    tailCuts.zip(tailCuts.tail).foreach { case (left, right) =>
      left.io.deq <> right.io.enq
    }
    tailCuts.last.io.deq <> io.out
  }
}

object DecoupledCutEmitter extends App {
  println(getVerilogString(new DecoupledCut(UInt(8.W), 2)))
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
        new DecoupledCut(chiselTypeOf(left.bits), delay = 1, pipeline = false)
      )
      buffer.suggestName("fullCut1")

      left <> buffer.io.in
      buffer.io.out <> right
    }

    def -||>(
        right: DecoupledIO[T]
    )(implicit sourceInfo: chisel3.experimental.SourceInfo): Unit = {
      val buffer = Module(
        new DecoupledCut(chiselTypeOf(left.bits), delay = 2, pipeline = false)
      )
      buffer.suggestName("fullCut2")

      left <> buffer.io.in
      buffer.io.out <> right
    }

    def -|||>(
        right: DecoupledIO[T]
    )(implicit sourceInfo: chisel3.experimental.SourceInfo): Unit = {
      val buffer = Module(
        new DecoupledCut(chiselTypeOf(left.bits), delay = 3, pipeline = false)
      )
      buffer.suggestName("fullCut3")

      left <> buffer.io.in
      buffer.io.out <> right
    }

    def -\>(
        right: DecoupledIO[T]
    )(implicit sourceInfo: chisel3.experimental.SourceInfo): Unit = {
      val buffer = Module(
        new DecoupledCut(chiselTypeOf(left.bits), delay = 1, pipeline = true)
      )
      buffer.suggestName("halfCut1")

      left <> buffer.io.in
      buffer.io.out <> right
    }

    def -\\>(
        right: DecoupledIO[T]
    )(implicit sourceInfo: chisel3.experimental.SourceInfo): Unit = {
      val buffer = Module(
        new DecoupledCut(chiselTypeOf(left.bits), delay = 2, pipeline = true)
      )
      buffer.suggestName("halfCut2")

      left <> buffer.io.in
      buffer.io.out <> right
    }

    def -\\\>(
        right: DecoupledIO[T]
    )(implicit sourceInfo: chisel3.experimental.SourceInfo): Unit = {
      val buffer = Module(
        new DecoupledCut(chiselTypeOf(left.bits), delay = 3, pipeline = true)
      )
      buffer.suggestName("halfCut3")

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
