package snax.simd

import chisel3._
import chisel3.util._

import snax.DataPathExtension.DataPathExtension
import snax.DataPathExtension.DataPathExtensionParam

/** The SIMD core's operators: one lane per beat element, one beat per cycle.
  *
  * The beat is the TCDM port width, `num_channel` x 64 bits, so the lane count of every operator is set by the port
  * count and nothing else: `lanes = num_channel * 64 / elementWidth`. Each lane has its own FP units; nothing is
  * time-shared across lanes and there is no per-operator lane knob. An operator is
  *
  * {{{
  *   input register -> per-lane pipeline (fixed latency, no stall input) -> SimdOutputStage (FIFO + credit)
  * }}}
  *
  * and its control is what that structure needs and no more: a valid bit travelling down the pipeline, the credit that
  * admits a beat only when its result has a FIFO slot, and whatever row bookkeeping the operator's CSRs ask for.
  */
abstract class SimdStreamOp(implicit extensionParam: DataPathExtensionParam) extends DataPathExtension {

  /** A valid bit delayed `n` cycles and dropped by `ext_start_i`, so a result still in flight when the next task starts
    * never reaches the FIFO after the credit has been reset.
    */
  def validPipe(in: Bool, n: Int): Bool =
    if (n <= 0) in
    else {
      val r = RegInit(VecInit(Seq.fill(n)(false.B)))
      r(0) := Mux(ext_start_i, false.B, in)
      for (i <- 1 until n) r(i) := Mux(ext_start_i, false.B, r(i - 1))
      r(n - 1)
    }
}

object SimdStreamOp {

  /** Balanced pairwise reduction: each level combines element 2i with 2i+1, and an odd element out waits one level in a
    * register. `node` must carry exactly `lat` cycles of latency, so every operand of a level arrives together and the
    * result is valid `ceil(log2(xs.length)) * lat` cycles after the inputs.
    */
  def treeReduce(xs: Seq[UInt], lat: Int)(node: (UInt, UInt) => UInt): UInt =
    if (xs.length == 1) xs.head
    else {
      val next = xs
        .grouped(2)
        .map {
          case Seq(a, b) => node(a, b)
          case Seq(a)    => ShiftRegister(a, lat)
        }
        .toSeq
      treeReduce(next, lat)(node)
    }

  def treeLevels(n: Int): Int = log2Ceil(n)
}

/** The output side shared by every SIMD operator.
  *
  * An operator's FP pipeline has no stall input: a beat it accepts leaves the pipeline `latency` cycles later whether
  * or not the consumer is ready. The FIFO holds those results, and the credit counter admits a beat only when a FIFO
  * slot is reserved for what it will produce, so the FIFO cannot overflow.
  *
  * DEPTH. A slot is held from the accept of an entry's first input beat until the credit register sees the entry leave:
  * `beatsPerEntry - 1 + latency` cycles to enqueue, one to dequeue (the FIFO does not flow through), one for the
  * counter. With a new entry starting every `beatsPerEntry` cycles that is `1 + ceil((latency + 1) / beatsPerEntry)`
  * entries outstanding -- the depth at which a consumer that is always ready sees one input beat per cycle. A shallower
  * FIFO is still correct; it only stalls the input.
  *
  * `reserve` is how many slots the operator claims this cycle (a reduce in tap mode claims two at a row start), and the
  * operator must not claim more than `credit`.
  */
class SimdOutputStage(width: Int, latency: Int, beatsPerEntry: Int = 1, maxReserve: Int = 1)
    extends Module
    with RequireAsyncReset {
  val depth                = 1 + (latency + beatsPerEntry) / beatsPerEntry
  override def desiredName = s"SimdOutputStage_w${width}_d${depth}_r$maxReserve"

  val io = IO(new Bundle {
    val clear   = Input(Bool()) // a new task: the stage is empty by contract, so only the credit resets
    val reserve = Input(UInt(log2Ceil(maxReserve + 1).W))
    val credit = Output(UInt(log2Ceil(depth + 1).W)) // free slots nobody has reserved
    val enq    = Flipped(Valid(UInt(width.W)))
    val deq    = Decoupled(UInt(width.W))
    val idle   = Output(Bool())                      // no slot reserved, nothing in flight or queued
  })

  private val fifo = Module(new Queue(UInt(width.W), depth))
  fifo.io.enq.valid := io.enq.valid
  fifo.io.enq.bits  := io.enq.bits
  assert(!io.enq.valid || fifo.io.enq.ready, "SimdOutputStage: a result arrived without a reserved slot")
  io.deq <> fifo.io.deq

  private val credit = RegInit(depth.U(log2Ceil(depth + 1).W))
  when(io.clear) {
    credit := depth.U
  }.otherwise {
    credit := credit - io.reserve + fifo.io.deq.fire.asUInt
  }
  io.credit := credit
  io.idle := credit === depth.U
}
