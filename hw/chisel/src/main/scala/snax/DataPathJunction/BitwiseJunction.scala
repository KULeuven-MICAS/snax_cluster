package snax.DataPathJunction

import chisel3._
import chisel3.util._

/** \============================================================================================================
  * `BitwiseJunction` -- the operator with no arithmetic at all.
  * \============================================================================================================
  *
  * The beat is 512 independent bits and the combine is one gate per bit:
  * {{{
  *   OR    (Z_2^n, or)    identity = all zeros      idempotent
  *   AND   (Z_2^n, and)   identity = all ones       idempotent
  *   XOR   (Z_2^n, xor)   identity = all zeros      self-inverse, NOT idempotent
  * }}}
  * All three are commutative monoids, so all three are legal collectives. `OR` is the union of Bloom filters and of
  * presence bitmaps, `AND` their intersection, and `XOR` is parity -- the reduction an erasure code or an
  * additively-shared secret is aggregated with.
  *
  * WHY IT IS HERE. Every other operator on this socket was designed around a datapath: the twisted family around an
  * exponential and a fused multiply-add, the linear operator around a lane grid, top-k around a comparator network.
  * This one has none of those. It contains no adder, no multiplier, no comparator and no state beyond a single retiming
  * stage, and it still satisfies the same four obligations. If the contract were secretly a description of the monoid
  * cell, this operator could not exist.
  *
  * ---- HOW IT MEETS THE CONTRACT ----
  * {{{
  *   O1 declared fixed latency   one registered stage, published to the chassis
  *   O2 format closure           a raw beat in, a raw beat out; there is no format to be closed under
  *   O3 identity tolerance       the identity is a WHOLE-BEAT CONSTANT chosen by the op, not a per-slot pad --
  *                               the only operator here whose identity has no lane structure at all
  *   O4 no state between pairs   one pipeline register; nothing survives a pair
  * }}}
  *
  * `XOR` is worth one line of care. It is a legal collective -- associative, commutative, with identity 0 -- but it is
  * NOT idempotent, so unlike `OR` and `AND` a duplicated beat corrupts it. An operator declaring duplicate-tolerance
  * would have to declare it per op, not per operator.
  *
  * ---- CSR(0) ----
  * {{{
  *   [1:0]  op   0 = OR, 1 = AND, 2 = XOR
  * }}}
  */
class HasBitwiseJunction(
  dataWidth:   Int = 512,
  skidDepth:   Int = 4,
  starveLimit: Int = 4096
) extends HasDataPathJunction {
  implicit val junctionParam: JunctionParam =
    new JunctionParam(
      moduleName  = "BitwiseJunction",
      userCsrNum  = 1,
      dataWidth   = dataWidth,
      starveLimit = starveLimit
    )

  def instantiate(clusterName: String): BitwiseJunction =
    Module(new BitwiseJunction(skidDepth = skidDepth) {
      override def desiredName = clusterName + namePostfix
    })
}

object BitwiseJunction {
  val OP_OR  = 0
  val OP_AND = 1
  val OP_XOR = 2
}

class BitwiseJunction(
  skidDepth:     Int = 4
)(implicit
  junctionParam: JunctionParam
) extends DataPathJunction {

  import BitwiseJunction._

  val opcode = jct_csr_i(0)(1, 0)

  val aQ = Module(new Queue(UInt(junctionParam.dataWidth.W), entries = skidDepth, pipe = true, flow = false))
  val bQ = Module(new Queue(UInt(junctionParam.dataWidth.W), entries = skidDepth, pipe = true, flow = false))
  aQ.io.enq <> jct_a_i
  bQ.io.enq <> jct_b_i

  /** O1: one registered stage. There is nothing to pipeline deeper -- the critical path is a single gate. */
  val latency = 1
  val Qdepth  = scala.math.max(2, latency + 4)
  val outQ    = Module(new Queue(UInt(junctionParam.dataWidth.W), entries = Qdepth))
  val credit  = RegInit(Qdepth.U(log2Ceil(Qdepth + 1).W))

  jct_cfgerr_o := opcode > OP_XOR.U // O5: only three of the four encodable ops exist

  // O3 strengthened, and the clearest case for publishing it rather than assuming it: AND's identity is all
  // ONES. A chassis that filled a missing operand with zeros -- the obvious default, and right for OR and XOR --
  // would zero every AND fold in the chain and produce a perfectly well-formed empty intersection.
  jct_identity_o := Mux(opcode === OP_AND.U, ~0.U(junctionParam.dataWidth.W), 0.U)

  val fire = aQ.io.deq.valid && bQ.io.deq.valid && (credit =/= 0.U) && !jct_start_i
  aQ.io.deq.ready := fire
  bQ.io.deq.ready := fire

  val a       = aQ.io.deq.bits
  val b       = bQ.io.deq.bits
  val outBeat = RegNext(
    MuxLookup(opcode, a | b)(
      Seq(OP_OR.U -> (a | b), OP_AND.U -> (a & b), OP_XOR.U -> (a ^ b))
    )
  )

  def clrPipe(in: Bool, n: Int): Bool =
    if (n <= 0) in
    else {
      val r = RegInit(VecInit(Seq.fill(n)(false.B)))
      r(0) := Mux(jct_start_i, false.B, in)
      for (i <- 1 until n) r(i) := Mux(jct_start_i, false.B, r(i - 1))
      r(n - 1)
    }
  val retire = clrPipe(fire, latency)

  outQ.io.enq.valid := retire && !jct_start_i
  outQ.io.enq.bits  := outBeat
  assert(!outQ.io.enq.valid || outQ.io.enq.ready, "BitwiseJunction: output queue overflow (credit bug)")
  outQ.io.deq.ready := jct_data_o.ready
  jct_data_o.valid  := outQ.io.deq.valid
  jct_data_o.bits   := outQ.io.deq.bits

  val deq = outQ.io.deq.fire
  when(jct_start_i) { credit := Qdepth.U }.otherwise {
    when(fire =/= deq) { credit := Mux(fire, credit - 1.U, credit + 1.U) }
  }

  jct_busy_o := (credit =/= Qdepth.U) || aQ.io.deq.valid || bQ.io.deq.valid
}
