package snax.utils

import chisel3._
import chisel3.util._

class AsyncQueue[T <: Data](dataType: T, depth: Int = 2)
    extends Module
    with RequireAsyncReset {
  require(
    isPow2(depth),
    "AsyncQueue requires perfect overflow with only one bits flipping"
  )
  val io = IO(new Bundle {
    val enq = new Bundle {
      val clock = Input(Clock())
      val data = Flipped(Decoupled(dataType))
    }
    val deq = new Bundle {
      val clock = Input(Clock())
      val data = Decoupled(dataType)
    }
  })

  // The bit is precalculated and will be used by both readCounter and writeCounter
  val counterBits = log2Up(depth)

  val readerPointer = Wire(UInt(counterBits.W))
  val writerPointer = Wire(UInt(counterBits.W))
  val readerPointerGray = Wire(UInt(counterBits.W))
  val writerPointerGray = Wire(UInt(counterBits.W))
  val nextReaderPointerGray = Wire(UInt(counterBits.W))
  val nextWriterPointerGray = Wire(UInt(counterBits.W))

  // empty and full signals will be driven later
  val empty = Wire(Bool())
  val full = Wire(Bool())
  io.enq.data.ready := ~full
  io.deq.data.valid := ~empty

  // The memory array for the FIFO
//   val mem = withClock(io.enq.clock) { Mem(depth, dataType) }
  val mem = Mem(depth, dataType)

  withClock(io.enq.clock) {
    val writerCounter = Counter(depth)
    writerPointer := writerCounter.value
    writerPointerGray := writerPointer ^ (writerPointer >> 1.U)
    nextWriterPointerGray := (writerPointer + 1.U) ^ ((writerPointer + 1.U) >> 1.U)

    full := nextWriterPointerGray === RegNext(
      RegNext(readerPointerGray, 0.U),
      0.U
    )

    when(io.enq.data.fire) {
      mem.write(writerPointer, io.enq.data.bits, io.enq.clock)
      writerCounter.inc()
    }
  }

  withClock(io.deq.clock) {
    val readerCounter = Counter(depth)
    readerPointer := readerCounter.value
    readerPointerGray := readerPointer ^ (readerPointer >> 1.U)
    nextReaderPointerGray := (readerPointer + 1.U) ^ ((readerPointer + 1.U) >> 1.U)

    empty := readerPointerGray === RegNext(
      RegNext(writerPointerGray, 0.U),
      0.U
    )

    when(io.deq.data.fire) {
      readerCounter.inc()
    }
  }

  io.deq.data.bits := mem.read(readerPointer, io.deq.clock)
}

object AsyncQueueEmitter extends App {
  emitVerilog(new AsyncQueue(UInt(8.W), 16), Array("--target-dir", "generated"))
}
