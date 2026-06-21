package snax.DataPathExtension

import chisel3._
import chisel3.util._

/** FpRecipRsqrt: the shared NR scalar primitive — 1/x (RECIP) and 1/sqrt(x) (RSQRT) on a single FP32
  * scalar, combinational. It is wrapped as a DataPathExtension by `StreamScalar` (StreamScalar.scala) and
  * is also reused per-instance inside `StreamMap` (operandMode). It is NOT a DataPathExtension itself.
  *
  * Newton-Raphson with a bit-trick seed. This is used instead of the
  * fpnew_divsqrt_th_32 (E906) blackbox because that drop-in pulls in fpnew_pkg + 12 vendored E906 .v
  * files + common_cells includes + package-typed ports — too deep to wire reliably overnight. NR gives
  * ~machine-precision after 4 iterations (recip is exact in practice; rsqrt within ~1 ULP).
  *
  *   recip(x): y0 = reinterpret(0x7EF127EA - bits(x)); y = y*(2 - x*y)        x4
  *   rsqrt(x): y0 = reinterpret(0x5F3759DF - (bits(x)>>1)); y = y*(1.5 - 0.5*x*y*y) x4
  */
// `modes` selects which NR path(s) to build (a subset of {RECIP, RSQRT}); only the listed loop(s) are
// instantiated. Defaults to both (used by StreamMap's shared operand unit, which needs recip AND rsqrt).
class FpRecipRsqrt(modes: Seq[String] = Seq("RECIP", "RSQRT")) extends Module with RequireAsyncReset {
  require(
    modes.nonEmpty && modes.toSet.subsetOf(Set("RECIP", "RSQRT")),
    s"FpRecipRsqrt: modes must be a non-empty subset of {RECIP, RSQRT}, got $modes"
  )
  val hasRecip = modes.contains("RECIP")
  val hasRsqrt = modes.contains("RSQRT")

  val io = IO(new Bundle {
    val in   = Input(UInt(32.W))  // FP32
    val mode = Input(UInt(1.W))   // 0 = RECIP, 1 = RSQRT (ignored if only one path is built)
    val out  = Output(UInt(32.W)) // FP32
  })

  import FpHelpers._

  val x = io.in

  // ---- reciprocal: y = y*(2 - x*y) ----
  val rY: UInt = if (hasRecip) {
    val TWO   = f32lit(2.0f)
    val rSeed = 0x7ef127ea.U(32.W) - x
    def recipIter(y: UInt): UInt = fmul(y, ffma(Cat(~x(31), x(30, 0)), y, TWO)) // y*( -x*y + 2 )
    (0 until 4).foldLeft(rSeed)((y, _) => recipIter(y))
  } else 0.U(32.W)

  // ---- inverse sqrt: y = y*(1.5 - 0.5*x*y*y) ----
  val sY: UInt = if (hasRsqrt) {
    val HALF  = f32lit(0.5f)
    val ONEP5 = f32lit(1.5f)
    val sSeed = 0x5f3759df.U(32.W) - (x >> 1)
    val halfX = fmul(HALF, x)
    def rsqrtIter(y: UInt): UInt = {
      val y2 = fmul(y, y)
      fmul(y, ffma(Cat(~halfX(31), halfX(30, 0)), y2, ONEP5)) // y*( -0.5x*y^2 + 1.5 )
    }
    (0 until 4).foldLeft(sSeed)((y, _) => rsqrtIter(y))
  } else 0.U(32.W)

  io.out :=
    (if (hasRecip && hasRsqrt) Mux(io.mode === 1.U, sY, rY)
     else if (hasRsqrt) sY
     else rY)
}
