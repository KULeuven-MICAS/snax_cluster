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

  /** Sweep exp over [-88, 8] and assert the worst FP16-narrowed ULP. `latency` accounts for the
    * pipelined datapath (input is held constant across `latency`+1 cycles so the pipe fills with it). */
  def sweep(dut: FpExp, latency: Int, maxUlpAllowed: Int, tag: String): Unit = {
    dut.clock.step(1)
    var maxUlp = 0
    var worstX = 0.0f
    var maxRel = 0.0
    val xs = (-880 to 80).map(_ * 0.1f) // exp(8) ~ 2981 < 65504 stays in FP16 range
    for (x <- xs) {
      dut.io.in.poke(fbits(x).U)
      dut.clock.step(latency + 1)
      val hw     = asF(dut.io.out.peekInt())
      val golden = math.exp(x.toDouble).toFloat
      val ulp    = math.abs(f32ToF16bits(hw) - f32ToF16bits(golden))
      if (ulp > maxUlp) { maxUlp = ulp; worstX = x }
      if (golden != 0.0f) {
        val rel = math.abs((hw - golden) / golden).toDouble
        if (rel > maxRel) maxRel = rel
      }
    }
    println(f"[FpExp:$tag] worst FP16 ULP=$maxUlp at x=$worstX%.2f ; worst FP32 rel-err=$maxRel%.3e")
    assert(maxUlp <= maxUlpAllowed, s"FpExp($tag) exceeds $maxUlpAllowed FP16 ULP: $maxUlp at x=$worstX")
  }

  "FpExp" should "match host exp within <=1 FP16 ULP after narrowing" in {
    test(new FpExp)
      .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) {
        dut => sweep(dut, latency = 0, maxUlpAllowed = 1, tag = "comb256")
      }
  }

  // Area config used by StreamMap: pipelined + 128-entry ROM must stay within the relaxed <=2 FP16 ULP.
  "FpExp_pipelined_lut128" should "match host exp within <=2 FP16 ULP (pipelined, 128-entry ROM)" in {
    test(new FpExp(pipelined = true, lutN = 128))
      .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) {
        dut => sweep(dut, latency = FpExp.PipeLatency, maxUlpAllowed = 2, tag = "pipe128")
      }
  }
}
