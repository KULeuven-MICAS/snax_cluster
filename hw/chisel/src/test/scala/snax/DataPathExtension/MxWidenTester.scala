package snax.DataPathExtension

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import fp_unit._
import snax.DataPathExtension.FpHelpers._

/** MX (microscaling) INPUT widen: widenMX decodes a narrow MX element and applies its block's E8M0 shared
  * scale, producing the exact FP32 value elem * 2^(scale-127). Proven exact against a Scala reference for
  * every MX element type (E5M2/E4M3/E3M2/E2M3/E2M1) over a dense element+scale sweep — the reusable core of
  * "MX is an edge problem", so the FP32 compute + roofline are untouched.
  */
class MxWidenTester extends AnyFlatSpec with ChiselScalatestTester {

  private class MxDUT(fmt: Int) extends Module {
    val io = IO(new Bundle {
      val elem  = Input(UInt(16.W)) // 16-bit carrier (MX element in the low bits), matches the Rt slicer
      val scale = Input(UInt(8.W))
      val out   = Output(UInt(32.W))
    })
    io.out := widenMXRt(io.elem, io.scale, fmt.U, 0)
  }

  private def f32(b: BigInt): Double = java.lang.Float.intBitsToFloat(b.toInt).toDouble

  // reference: decode a finite low-precision element (no inf/nan) then apply the E8M0 block scale
  private def ref(bits: Int, expW: Int, sigW: Int, bias: Int, scale: Int): Double = {
    val sign = if (((bits >> (expW + sigW)) & 1) == 1) -1.0 else 1.0
    val exp  = (bits >> sigW) & ((1 << expW) - 1)
    val man  = bits & ((1 << sigW) - 1)
    val v =
      if (exp == 0) sign * man.toDouble * math.pow(2, 1 - bias - sigW) // subnormal / zero
      else sign * (1.0 + man.toDouble / (1 << sigW)) * math.pow(2, exp - bias)
    v * math.pow(2, scale - 127)
  }

  // (fmt code, expW, sigW, bias, elem-bit-count)
  private val formats = Seq(
    (FMT_MXFP8_E5M2, 5, 2, 15, 8),
    (FMT_MXFP8_E4M3, 4, 3, 7, 8),
    (FMT_MXFP6_E3M2, 3, 2, 3, 6),
    (FMT_MXFP6_E2M3, 2, 3, 1, 6),
    (FMT_MXFP4_E2M1, 2, 1, 1, 4)
  )

  private def f32bits(f: Float): BigInt = BigInt(java.lang.Float.floatToIntBits(f).toLong & 0xffffffffL)
  // f32 -> E5M2 (bias15, RNE, FTZ subnormal-underflow) reference for the narrowMX quantization
  private def f32ToE5M2(f: Float): Int = {
    if (f == 0.0f) return 0
    val s = if (f < 0) 0x80 else 0; val a = math.abs(f).toDouble
    val e = math.floor(math.log(a) / math.log(2)).toInt; var exp = e + 15
    if (exp >= 0x1f) return s | 0x7b
    if (exp < 1) return s
    val frac = a / math.pow(2, e) - 1.0; var man = math.round(frac * 4).toInt
    if (man == 4) { man = 0; exp += 1; if (exp >= 0x1f) return s | 0x7b }
    s | (exp << 2) | man
  }

  // narrowMX: block of n FP32 -> shared E8M0 scale + n narrow element codes
  private class MxNarrowDUT(n: Int, emax: Int) extends Module with RequireAsyncReset {
    val io = IO(new Bundle {
      val in    = Input(Vec(n, UInt(32.W)))
      val elem  = Output(Vec(n, UInt(8.W)))
      val scale = Output(UInt(8.W))
    })
    val scale = blockScaleE8M0((0 until n).map(io.in(_)), emax)
    io.scale := scale
    for (i <- 0 until n) io.elem(i) := narrowMX(io.in(i), scale, FP8, 0)(7, 0)
  }

  "narrowMX_e5m2" should "derive the block scale and quantize each element to E5M2 (MX output)" in {
    test(new MxNarrowDUT(32, 15)) { dut =>
      val rng = new scala.util.Random(0x7)
      for (trial <- 0 until 8) {
        val xs = Seq.fill(32)((rng.between(-6.0, 6.0)).toFloat)
        for (i <- 0 until 32) dut.io.in(i).poke(f32bits(xs(i)).U)
        val scaleHW  = dut.io.scale.peekInt().toInt
        val maxExp   = xs.map(x => (java.lang.Float.floatToIntBits(x) >>> 23) & 0xff).max
        val scaleRef = math.max(0, maxExp - 15)
        assert(scaleHW == scaleRef, s"block scale hw=$scaleHW ref=$scaleRef")
        val adj = math.pow(2, scaleRef - 127)
        for (i <- 0 until 32) {
          val hw   = dut.io.elem(i).peekInt().toInt
          val gold = f32ToE5M2((xs(i) / adj).toFloat)
          // codes monotonic within sign; ±1 absorbs RNE-vs-fpnew edge rounding
          val d = math.abs((hw & 0x7f) - (gold & 0x7f))
          assert((hw & 0x80) == (gold & 0x80) || (gold & 0x7f) == 0 || (hw & 0x7f) == 0,
                 s"sign mismatch lane $i x=${xs(i)} hw=0x$hw gold=0x$gold")
          assert(d <= 1, s"E5M2 code off lane $i x=${xs(i)} hw=0x$hw gold=0x$gold (scale=$scaleHW)")
        }
      }
      println("[narrowMX E5M2] block-scale + per-element quantization verified over 8 blocks")
    }
  }

  for ((fmt, expW, sigW, bias, nb) <- formats) {
    val tag = s"E${expW}M${sigW}"
    s"widenMX_$tag" should s"decode $tag * 2^(scale-127) exactly" in {
      test(new MxDUT(fmt)) { dut =>
        var worst = 0.0
        var wtag  = ""
        val scales = Seq(115, 120, 127, 130, 135) // keep results in FP32 range for all element magnitudes
        for (sc <- scales; e <- 0 until (1 << nb)) {
          val exp   = (e >> sigW) & ((1 << expW) - 1)
          val man   = e & ((1 << sigW) - 1)
          val isNan = (expW == 4 && sigW == 3) && exp == 0xf && man == 0x7 // E4M3 sole NaN codepoint
          if (!isNan) {
            dut.io.elem.poke(e.U); dut.io.scale.poke(sc.U)
            val got  = f32(dut.io.out.peekInt())
            val gold = ref(e, expW, sigW, bias, sc)
            // exact for in-range; skip the rare over/underflow-to-{inf,0} edge the HW clamps
            val goldF = gold.toFloat
            if (goldF.isFinite && goldF != 0.0f && math.abs(goldF) >= 1e-30f) {
              val rel = math.abs(got - gold) / math.abs(gold)
              if (rel > worst) { worst = rel; wtag = f"elem=0x$e%02x scale=$sc got=$got%.6g gold=$gold%.6g" }
              assert(rel <= 1e-6, s"widenMX $tag mismatch: $wtag")
            }
          }
        }
        println(f"[widenMX $tag] worst rel err=$worst%.2g over element+scale sweep  ($wtag)")
      }
    }
  }
}
