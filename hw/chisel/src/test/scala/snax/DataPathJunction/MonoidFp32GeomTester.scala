package snax.DataPathJunction

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

import snax.DataPathJunction.JunctionTestUtils._

/** FP32 TRANSPORT ON A 32-LANE INSTANCE, and the one thing software has to know about it.
  *
  * `elemWidth` fixes the lane count at elaboration, but `fmt` changes how many ELEMENTS a beat actually carries: 32
  * at FP16, 16 at FP32. The geometry saturation and the O5 check are both computed from the ELABORATION count, so a
  * word whose `F * S` exceeds the runtime element count is accepted and quietly means something else.
  *
  * This matters because FP32 transport is what lets the flash-attention triple be folded with NO re-embedding -- the
  * raw numerator fits FP32 with room to spare, so the key stays `m` and the values stay raw. The price is one partial
  * per beat instead of two, and it is a price only if `F * S <= 16` is respected.
  */
class MonoidFp32GeomTester extends AnyFlatSpec with ChiselScalatestTester {
  val flags = VerilatorFlags(Seq("--build-jobs", "1"))
  private val FP32 = ElementwiseJunction.FMT_FP32
  private def word(n: Int, nExp: Int, sigma: Int, nValid: Int, fmt: Int = FP32): BigInt =
    (BigInt(sigma) << 26) | (BigInt(nExp) << 18) | (BigInt(fmt) << 12) | (BigInt(n) << 8) | BigInt(nValid)

  /** one partial: key `k` on field 0, then v*j on field j -- laid out at sigma = 0, so field == lane */
  private def partial(k: Double, v: Double): BigInt = {
    var b = f32(k)
    for (j <- 1 until 16) b |= f32(v * j) << (32 * j)
    b
  }

  "MonoidJunction_fp32_triple" should "fold a 16-field partial in FP32 with no re-embedding" in {
    // F = 16, sigma = 0 -> S = 1 -> exactly the 16 elements an FP32 beat carries. This is the flash-attention
    // triple's chunk geometry when the numerator crosses in FP32: no O/l, no ln, no power-of-two prescale.
    test(new DataPathJunctionHarness(new HasMonoidJunction(elemWidth = 16)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val w = word(n = 15, nExp = 15, sigma = 0, nValid = 1)
        // equal keys => alpha = 1 on both sides => every value field is a plain sum
        val out = runPair(dut, w, partial(4.0, 1.0), partial(4.0, 2.0))
        assert(laneF32(out, 0) == 4.0, s"key lane: ${laneF32(out, 0)}")
        for (j <- 1 until 16)
          assert(laneF32(out, j) == 3.0 * j, s"field $j: got ${laneF32(out, j)}, want ${3.0 * j}")
        // and a RAW flash numerator, the magnitude that overflows FP16 by 48x, is carried exactly
        val big = 3175000.0
        val o2  = runPair(dut, word(n = 1, nExp = 1, sigma = 0, nValid = 1),
                          f32(4.0) | (f32(big) << 32), f32(4.0) | (f32(big) << 32))
        assert(laneF32(o2, 1) == 2 * big, s"raw numerator: got ${laneF32(o2, 1)}, want ${2 * big}")
        println("[fp32] F=16 sigma=0 folds exactly, and a 3.2e6 numerator crosses untouched")
      }
  }

  "MonoidJunction_fp32_laneBudget" should "document that F*S > 16 at FP32 is NOT caught by O5" in {
    // THE HAZARD. `F = 16, sigma = 1` needs 32 lanes. That is legal at FP16 and is exactly the word the FP16
    // layout uses -- but an FP32 beat has only 16 elements, so slot 1 reads slot 0's fields. The result is
    // finite, format-legal and wrong, and `jct_cfgerr_o` stays LOW because the check uses the elaboration lane
    // count. Software must keep F * S <= dataWidth / 32 when fmt = FP32.
    //
    // If this assertion ever flips, someone has taught O5 the runtime element count -- good; update it.
    test(new DataPathJunctionHarness(new HasMonoidJunction(elemWidth = 16)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val bad = word(n = 15, nExp = 15, sigma = 1, nValid = 2)
        dut.io.csr_i(0).poke(bad.U); dut.io.enable_i.poke(true); dut.clock.step(2)
        assert(!dut.io.cfgerr_o.peekBoolean(), "O5 now catches the FP32 lane-budget overrun -- update this test")
        val ok  = runPair(dut, word(n = 15, nExp = 15, sigma = 0, nValid = 1), partial(4.0, 1.0), partial(4.0, 2.0))
        val out = runPair(dut, bad, partial(4.0, 1.0), partial(4.0, 2.0))
        assert(out != ok, "the over-wide geometry should not coincidentally agree with the legal one")
        println("[fp32] F*S = 32 at FP32 silently truncates to the low 16 elements; O5 does not flag it")
      }
  }
}
