package snax.DataPathExtension

import chisel3._

import fp_unit._
import fp_native._

/** Shared FP primitives for the SIMD xDMA extensions. FP16 transport, FP32 internal.
  *
  * The native-Chisel FP units (fp_native.FpAdd/FpMul/FpFma, 1:1 ports of fpnew fp_add/fp_mul/fp_fma)
  * support mixed in/out formats, so widen and narrow are just adds with a zero operand. fp32max/min are
  * pure Chisel (no NaN handling — the host softmax path produces only finite values). Each arithmetic /
  * widen / narrow call instantiates one unit at the call site (legal inside a Module body).
  */
object FpHelpers {
  // FP32 / FP16 bit-pattern literals computed at elaboration
  def f32lit(v: Float): UInt = BigInt(java.lang.Float.floatToIntBits(v).toLong & 0xffffffffL).U(32.W)
  def f16lit(bits: Int): UInt = (bits & 0xffff).U(16.W)

  val FP32_ZERO = 0.U(32.W)
  val FP16_ZERO = 0.U(16.W)

  // Combinational FP32 max / min for finite operands.
  def fp32max(a: UInt, b: UInt): UInt = {
    val sa = a(31); val sb = b(31)
    Mux(Mux(sa =/= sb, !sa, Mux(sa, b >= a, a >= b)), a, b)
  }
  def fp32min(a: UInt, b: UInt): UInt = {
    val sa = a(31); val sb = b(31)
    Mux(Mux(sa =/= sb, sa, Mux(sa, a >= b, b >= a)), a, b)
  }

  // FP32 arithmetic (a*b+c etc.) via the native-Chisel FP units. `numPipe` = internal pipeline depth of
  // the FP unit (the per-op "cutting" knob for timing; 0 = combinational). The host FSM accounts for it.
  def fadd(a: UInt, b: UInt, numPipe: Int = 0): UInt = {
    val m = Module(new FpAdd(FP32, FP32, FP32, numPipe)); m.io.in_a := a; m.io.in_b := b; m.io.out
  }
  def fmul(a: UInt, b: UInt, numPipe: Int = 0): UInt = {
    val m = Module(new FpMul(FP32, FP32, FP32, numPipe)); m.io.in_a := a; m.io.in_b := b; m.io.out
  }
  def ffma(a: UInt, b: UInt, c: UInt, numPipe: Int = 0): UInt = {
    val m = Module(new FpFma(FP32, FP32, FP32, numPipe)); m.io.in_a := a; m.io.in_b := b; m.io.in_c := c; m.io.out
  }

  // transport-type <-> FP32 conversions (exact widen; RNE narrow), via add-with-zero in the target format.
  // `t` is the FP16/BF16/FP8/FP32 transport (element) type; the internal compute stays FP32.
  def widen(h: UInt, t: FpType, numPipe: Int = 0): UInt = {
    val m = Module(new FpAdd(t, t, FP32, numPipe)); m.io.in_a := h; m.io.in_b := 0.U(t.width.W); m.io.out
  }
  def narrow(f: UInt, t: FpType, numPipe: Int = 0): UInt = {
    val m = Module(new FpAdd(FP32, FP32, t, numPipe)); m.io.in_a := f; m.io.in_b := FP32_ZERO; m.io.out
  }
  def square(h: UInt, t: FpType, numPipe: Int = 0): UInt = { // t*t -> FP32
    val m = Module(new FpMul(t, t, FP32, numPipe)); m.io.in_a := h; m.io.in_b := h; m.io.out
  }

  // FP16 <-> FP32 (the legacy fixed-precision aliases)
  def widenF16(h: UInt): UInt  = widen(h, FP16)
  def narrowF32(f: UInt): UInt = narrow(f, FP16)
}

/** Parses the op-set list shared by the SIMD extensions. Each entry is "<OP>_<PRECISION>" where PRECISION
  * is the TRANSPORT (element) type the op processes — FP16 | BF16 | FP8 | FP32 (internal compute is always
  * FP32). All entries in one extension must share a single precision (one element width per stream). The
  * precision is configurable from the hjson and sets the datapath element width; only FP16 is currently
  * exercised by the apps, the others elaborate via the fpnew mixed-format blackboxes.
  */
object OpSpec {
  val precision: Map[String, FpType] = Map("FP16" -> FP16, "BF16" -> BF16, "FP8" -> FP8, "FP32" -> FP32)

  /** -> (bare op names, shared transport FpType). `allowedOps` validates the op names; `who` names the
    * caller for error messages. */
  def parse(entries: Seq[String], allowedOps: Set[String], who: String): (Seq[String], FpType) = {
    require(entries.nonEmpty, s"$who: op list must be non-empty")
    val split = entries.map { e =>
      val i = e.lastIndexOf('_')
      require(i > 0 && i < e.length - 1, s"$who: '$e' must be of the form <OP>_<PRECISION>")
      (e.substring(0, i), e.substring(i + 1))
    }
    val ops   = split.map(_._1)
    val precs = split.map(_._2).distinct
    require(ops.toSet.subsetOf(allowedOps), s"$who: ops must be a subset of $allowedOps, got $ops")
    require(precs.size == 1, s"$who: all ops must share one transport precision, got $precs")
    require(
      precision.contains(precs.head),
      s"$who: unknown precision '${precs.head}', known: ${precision.keys.mkString(", ")}"
    )
    (ops, precision(precs.head))
  }

  /** Validates an explicit config `elementWidth` against the width implied by the op-set precision (the
    * precision tag is the source of truth — FP16/BF16⇒16, FP8⇒8, FP32⇒32). The config carries the width
    * only to make it visible; a mismatch (e.g. elementWidth:8 with an FP16 op) is a config error. */
  def checkWidth(elementWidth: Int, transport: FpType, who: String): Unit =
    require(
      elementWidth == transport.width,
      s"$who: elementWidth=$elementWidth must equal the op precision width ${transport.width} ($transport)"
    )
}
