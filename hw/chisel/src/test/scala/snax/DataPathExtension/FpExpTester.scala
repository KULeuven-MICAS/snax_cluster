package snax.DataPathExtension

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

/** Tier-1 accuracy test for FpExp vs the host Cephes exp. Gating milestone: the FP16-narrowed result
  * must match within <=1 FP16 ULP across [-88, 8] (design-doc §5.1). Reports the worst-case ULP.
  */
class FpExpTester extends AnyFlatSpec with ChiselScalatestTester {

  def fbits(f: Float): BigInt = BigInt(java.lang.Float.floatToIntBits(f).toLong & 0xffffffffL)
  def asF(bits: BigInt): Float = java.lang.Float.intBitsToFloat(bits.toInt)

  def f32ToF16bits(f: Float): Int = {
    val bits = java.lang.Float.floatToIntBits(f)
    val sign = (bits >>> 16) & 0x8000
    val rawe = (bits >>> 23) & 0xff
    val mant = bits & 0x7fffff
    if (rawe == 0xff) return sign | 0x7c00 | (if (mant != 0) 0x200 else 0)
    val exp = rawe - 127 + 15
    if (exp >= 0x1f) return sign | 0x7c00
    if (exp <= 0) {
      if (exp < -10) return sign
      val m       = mant | 0x800000
      val shift   = 14 - exp
      val half    = m >>> shift
      val rem     = m & ((1 << shift) - 1)
      val halfway = 1 << (shift - 1)
      var r       = half
      if (rem > halfway || (rem == halfway && (half & 1) == 1)) r += 1
      return sign | r
    }
    var h       = sign | (exp << 10) | (mant >>> 13)
    val rem     = mant & 0x1fff
    val halfway = 0x1000
    if (rem > halfway || (rem == halfway && (h & 1) == 1)) h += 1
    h
  }

  "FpExp" should "match host exp within <=1 FP16 ULP after narrowing" in {
    test(new FpExp)
      .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) {
        dut =>
          dut.clock.step(1)
          var maxUlp   = 0
          var worstX   = 0.0f
          var maxRel   = 0.0
          // sweep [-88, 8]; exp stays within FP16 range (exp(8) ~ 2981 < 65504)
          val xs = (-880 to 80).map(_ * 0.1f)
          for (x <- xs) {
            dut.io.in.poke(fbits(x).U)
            dut.clock.step(1)
            val hw     = asF(dut.io.out.peekInt())
            val golden = math.exp(x.toDouble).toFloat
            val ulp    = math.abs(f32ToF16bits(hw) - f32ToF16bits(golden))
            if (ulp > maxUlp) { maxUlp = ulp; worstX = x }
            if (golden != 0.0f) {
              val rel = math.abs((hw - golden) / golden).toDouble
              if (rel > maxRel) maxRel = rel
            }
          }
          println(f"[FpExp] worst FP16 ULP=$maxUlp at x=$worstX%.2f ; worst FP32 rel-err=$maxRel%.3e")
          assert(maxUlp <= 1, s"FpExp exceeds 1 FP16 ULP: $maxUlp at x=$worstX")
      }
  }
}
