package snax.DataPathExtension

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

/** Tier-1 accuracy test for FpSilu vs the host silu, silu(x) = x*sigmoid(x). Gating milestone: the
  * FP16-narrowed result must match within <= 1 FP16 ULP across the active range (and the saturating
  * tails). Reports the worst-case ULP. Mirrors FpExpTester; silu is signed, so the ULP is measured on a
  * monotonic FP16 ordering key.
  */
class FpSiluTester extends AnyFlatSpec with ChiselScalatestTester {

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

  // monotonic FP16 ordering key (handles signed silu): adjacent FP16 values map to adjacent keys, so
  // |key(a) - key(b)| is the FP16-ULP distance even across zero.
  def f16mono(h: Int): Int = { val mag = h & 0x7fff; if ((h & 0x8000) != 0) 0x8000 - mag else 0x8000 + mag }

  "FpSilu" should "match host silu within <= 1 FP16 ULP after narrowing" in {
    test(new FpSilu)
      .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) {
        dut =>
          dut.clock.step(1)
          var maxUlp = 0
          var worstX = 0.0f
          // sweep [-20, 20]: covers the [-16,16] table, the curvy origin, and both saturating tails.
          // Skip FP16 subnormal/zero goldens (exp field 0): the deep negative tail (x < ~-12.5) is silu ~ 0
          // noise that the DM core's FTZ host path also discards, exactly as the softmax/rmsnorm apps do.
          val xs = (-200 to 200).map(_ * 0.1f)
          for (x <- xs) {
            dut.io.in.poke(fbits(x).U)
            dut.clock.step(1)
            val golden = (x.toDouble / (1.0 + math.exp(-x.toDouble))).toFloat
            val gBits  = f32ToF16bits(golden)
            if ((gBits & 0x7c00) != 0) { // significant (normal) golden only
              val hw  = asF(dut.io.out.peekInt())
              val ulp = math.abs(f16mono(f32ToF16bits(hw)) - f16mono(gBits))
              if (ulp > maxUlp) { maxUlp = ulp; worstX = x }
            }
          }
          println(f"[FpSilu] worst FP16 ULP=$maxUlp at x=$worstX%.2f (significant outputs)")
          assert(maxUlp <= 1, s"FpSilu exceeds 1 FP16 ULP: $maxUlp at x=$worstX")
      }
  }
}
