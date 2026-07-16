package snax.DataPathExtension

import scala.util.Random

import chisel3._
import chisel3.util._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

import fp_unit._

/** Proof-of-concept for RUNTIME-selectable precision: one elaborated block converts FP16 / BF16 / FP8 at
  * the edges by a runtime `fmt` mux, and must reproduce the trusted compile-time per-format widen/narrow
  * bit-exactly. This is the whole mechanism of runtime precision (internal compute is FP32, so only the
  * edges are format-specific). If this holds, a single netlist can process any of the three precisions.
  */
class RtCheckDUT extends Module with RequireAsyncReset {
  val io = IO(new Bundle {
    val raw16 = Input(UInt(16.W)) // transport bits (FP8 in low 8)
    val f32in = Input(UInt(32.W))
    val wRt   = Output(Vec(3, UInt(32.W))) // widenRt for fmt 0,1,2
    val wSt   = Output(Vec(3, UInt(32.W))) // compile-time widen FP16,BF16,FP8
    val nRt   = Output(Vec(3, UInt(16.W))) // narrowRt
    val nSt   = Output(Vec(3, UInt(16.W))) // compile-time narrow (FP8 zero-extended)
  })
  io.wRt(0) := FpHelpers.widenRt(io.raw16, FpHelpers.FMT_FP16.U)
  io.wRt(1) := FpHelpers.widenRt(io.raw16, FpHelpers.FMT_BF16.U)
  io.wRt(2) := FpHelpers.widenRt(io.raw16, FpHelpers.FMT_FP8.U)
  io.wSt(0) := FpHelpers.widen(io.raw16(15, 0), FP16)
  io.wSt(1) := FpHelpers.widen(io.raw16(15, 0), BF16)
  io.wSt(2) := FpHelpers.widen(io.raw16(7, 0), FP8)
  io.nRt(0) := FpHelpers.narrowRt(io.f32in, FpHelpers.FMT_FP16.U)
  io.nRt(1) := FpHelpers.narrowRt(io.f32in, FpHelpers.FMT_BF16.U)
  io.nRt(2) := FpHelpers.narrowRt(io.f32in, FpHelpers.FMT_FP8.U)
  io.nSt(0) := FpHelpers.narrow(io.f32in, FP16)
  io.nSt(1) := FpHelpers.narrow(io.f32in, BF16)
  io.nSt(2) := Cat(0.U(8.W), FpHelpers.narrow(io.f32in, FP8))
}

/** A full RUNTIME-precision MAC lane: widen the transport input to FP32 by `fmt`, do the FP32³ MAC a*x+b
  * (the format-agnostic core, unchanged), narrow back to transport by `fmt`. One netlist, any precision.
  * This is exactly the per-lane compute of a runtime-precision StreamMap (LINEAR). */
class MacLaneRtDUT extends Module with RequireAsyncReset {
  val io = IO(new Bundle {
    val x16 = Input(UInt(16.W)) // transport input (FP8 in low 8)
    val a   = Input(UInt(32.W)) // FP32 coefficient a
    val b   = Input(UInt(32.W)) // FP32 coefficient b
    val fmt = Input(UInt(2.W))
    val out = Output(UInt(16.W)) // transport output (FP8 in low 8)
  })
  val xF32 = FpHelpers.widenRt(io.x16, io.fmt)  // runtime widen -> FP32
  val res  = FpHelpers.ffma(io.a, xF32, io.b)   // FP32³ MAC: a*x + b (format-agnostic; unchanged)
  io.out := FpHelpers.narrowRt(res, io.fmt)     // runtime narrow -> transport
}

/** Full 512-bit beat mapped at RUNTIME precision — the "runtime lane count" crux: the SAME netlist packs
  * 32 FP16 elements (16b each) OR 64 FP8 elements (8b each) in one beat, sliced/packed by `fmt`, 64 physical
  * MAC lanes (at FP16 only 0..31 carry data). No streaming FSM — this isolates the slicer/lanes/packer. */
class MapBeatRtDUT extends Module with RequireAsyncReset {
  val maxLanes = 64
  val io = IO(new Bundle {
    val inBeat  = Input(UInt(512.W))
    val a       = Input(UInt(32.W))
    val b       = Input(UInt(32.W))
    val fmt     = Input(UInt(2.W))
    val outBeat = Output(UInt(512.W))
  })
  // slice: carrier(i) = FP8 byte (zero-ext) or FP16/BF16 halfword; lanes 32..63 inactive at 16-bit formats
  val outs = (0 until maxLanes).map { i =>
    val fp8  = Cat(0.U(8.W), io.inBeat(8 * i + 7, 8 * i))
    val f16  = if (i < 32) io.inBeat(16 * i + 15, 16 * i) else 0.U(16.W)
    val cr   = Mux(io.fmt === FpHelpers.FMT_FP8.U, fp8, f16)
    val xF32 = FpHelpers.widenRt(cr, io.fmt)             // runtime widen -> FP32
    val res  = FpHelpers.ffma(io.a, xF32, io.b)          // FP32³ MAC a*x+b (format-agnostic core)
    FpHelpers.narrowRt(res, io.fmt)                      // 16b carrier (FP8 in low 8)
  }
  val packed8  = Cat((0 until 64).map(i => outs(63 - i)(7, 0)))   // 64 bytes -> 512b (lane0 low)
  val packed16 = Cat((0 until 32).map(i => outs(31 - i)(15, 0)))  // 32 halfwords -> 512b
  io.outBeat := Mux(io.fmt === FpHelpers.FMT_FP8.U, packed8, packed16)
}

class RuntimePrecisionTester extends AnyFlatSpec with ChiselScalatestTester {

  val fmts = Seq("FP16", "BF16", "FP8")

  // FP16 <-> FP32 (for a numerical golden of the full MAC lane at FP16; FP8/BF16 are covered compositionally
  // by the bit-exact converter test + the shared FP32 core)
  def f16bitsToF32(h: Int): Float = {
    val sign = if ((h & 0x8000) != 0) -1.0 else 1.0
    val exp  = (h >> 10) & 0x1f
    val mant = h & 0x3ff
    (if (exp == 0) sign * mant * math.pow(2, -24)
     else if (exp == 0x1f) if (mant == 0) sign * Double.PositiveInfinity else Double.NaN
     else sign * (1024 + mant) * math.pow(2, exp - 25)).toFloat
  }
  def f32ToF16bits(f: Float): Int = {
    val bits = java.lang.Float.floatToIntBits(f); val sign = (bits >>> 16) & 0x8000
    val rawe = (bits >>> 23) & 0xff; val mant = bits & 0x7fffff
    if (rawe == 0xff) return sign | 0x7c00 | (if (mant != 0) 0x200 else 0)
    val exp = rawe - 127 + 15
    if (exp >= 0x1f) return sign | 0x7c00
    if (exp <= 0) { if (exp < -10) return sign
      val m = mant | 0x800000; val shift = 14 - exp; val half = m >>> shift
      val rem = m & ((1 << shift) - 1); val hw = 1 << (shift - 1)
      var r = half; if (rem > hw || (rem == hw && (half & 1) == 1)) r += 1; return sign | r }
    var h = sign | (exp << 10) | (mant >>> 13); val rem = mant & 0x1fff
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1)) h += 1; h
  }
  def f32bits(f: Float): BigInt = BigInt(java.lang.Float.floatToIntBits(f).toLong & 0xffffffffL)

  "runtime_mac_lane" should "compute a*x+b at runtime FP16 (same netlist that serves FP8/BF16)" in {
    test(new MacLaneRtDUT)
      .withAnnotations(Seq(VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        val rng = new Random(0x1D)
        val (a, b) = (2.0f, -1.5f)
        dut.io.a.poke(f32bits(a).U); dut.io.b.poke(f32bits(b).U); dut.io.fmt.poke(FpHelpers.FMT_FP16.U)
        var maxErr = 0.0f; var n = 0
        for (_ <- 0 until 500) {
          val xf   = rng.between(-4, 4) + rng.nextInt(4) * 0.25f
          val xh   = f32ToF16bits(xf)
          dut.io.x16.poke(xh.U); dut.clock.step(1)
          val got  = f16bitsToF32(dut.io.out.peekInt().toInt & 0xffff)
          val gold = f16bitsToF32(f32ToF16bits(a * f16bitsToF32(xh) + b))
          val e = math.abs(got - gold); if (e > maxErr) maxErr = e
          assert(e <= math.max(math.abs(gold) * 0.01f, 0.02f), s"MAC-lane FP16 x=$xf hw=$got gold=$gold")
          n += 1
        }
        println(f"[RuntimePrecision] runtime MAC lane a*x+b @FP16 over $n vectors, maxErr=$maxErr%.4g")
      }
  }

  "runtime_widen_known" should "widen 1.0/2.0/-1.0 in each format to the right FP32" in {
    test(new RtCheckDUT)
      .withAnnotations(Seq(VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        // (raw transport bits per format) -> expected FP32 bits.  FP16 1.0=0x3C00, BF16 1.0=0x3F80,
        // FP8(E5M2,bias15) 1.0=0x3C; 2.0 = 0x4000/0x4000/0x40; -1.0 = 0xBC00/0xBF80/0xBC.
        val cases = Seq(
          (Seq(0x3c00, 0x3f80, 0x3c), 0x3f800000L), // +1.0
          (Seq(0x4000, 0x4000, 0x40), 0x40000000L), // +2.0
          (Seq(0xbc00, 0xbf80, 0xbc), 0xbf800000L)  // -1.0
        )
        for ((raws, exp) <- cases; i <- 0 until 3) {
          dut.io.raw16.poke(raws(i).U)
          dut.clock.step(1)
          val got = dut.io.wRt(i).peekInt()
          assert(got == BigInt(exp & 0xffffffffL),
                 f"widenRt(${fmts(i)}) raw=0x${raws(i)}%x -> 0x$got%x, expected 0x$exp%x")
        }
        println("[RuntimePrecision] known-value widen (1.0/2.0/-1.0 x FP16/BF16/FP8) OK")
      }
  }

  "runtime_edge_converters" should "match the compile-time per-format widen/narrow bit-exactly" in {
    test(new RtCheckDUT)
      .withAnnotations(Seq(VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        val rng = new Random(0x9e)
        var n   = 0
        for (_ <- 0 until 3000) {
          val raw = BigInt(rng.nextInt(1 << 16))
          val f32 = BigInt(rng.nextInt()) & BigInt("ffffffff", 16) // arbitrary FP32 (incl inf/nan/subnormal)
          dut.io.raw16.poke(raw.U)
          dut.io.f32in.poke(f32.U)
          dut.clock.step(1)
          for (i <- 0 until 3) {
            assert(dut.io.wRt(i).peekInt() == dut.io.wSt(i).peekInt(),
                   s"widenRt(${fmts(i)}) != widen for raw=$raw")
            assert(dut.io.nRt(i).peekInt() == dut.io.nSt(i).peekInt(),
                   s"narrowRt(${fmts(i)}) != narrow for f32=$f32")
          }
          n += 1
        }
        println(s"[RuntimePrecision] widenRt & narrowRt bit-exact vs compile-time FP16/BF16/FP8 over $n vectors")
      }
  }

  "runtime_beat_fp16" should "map a full 32-lane FP16 beat (a*x+b) on the shared 64-lane netlist" in {
    test(new MapBeatRtDUT)
      .withAnnotations(Seq(VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        val rng = new Random(0x2B); val (a, b) = (2.0f, -1.5f)
        dut.io.a.poke(f32bits(a).U); dut.io.b.poke(f32bits(b).U); dut.io.fmt.poke(FpHelpers.FMT_FP16.U)
        var maxErr = 0.0f
        for (_ <- 0 until 200) {
          val xs   = Seq.fill(32)(f32ToF16bits(rng.between(-4, 4) + rng.nextInt(4) * 0.25f))
          val beat = xs.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (h, i)) => acc | (BigInt(h & 0xffff) << (16 * i)) }
          dut.io.inBeat.poke(beat.U); dut.clock.step(1)
          val out = dut.io.outBeat.peekInt()
          for (i <- 0 until 32) {
            val got  = f16bitsToF32(((out >> (16 * i)) & 0xffff).toInt)
            val gold = f16bitsToF32(f32ToF16bits(a * f16bitsToF32(xs(i)) + b))
            val e = math.abs(got - gold); if (e > maxErr) maxErr = e
            assert(e <= math.max(math.abs(gold) * 0.01f, 0.02f), s"FP16 beat lane $i hw=$got gold=$gold")
          }
        }
        println(f"[RuntimePrecision] full 32-lane FP16 beat map OK, maxErr=$maxErr%.4g")
      }
  }

  "runtime_beat_identity" should "round-trip a full beat at FP8 (64 lanes) and FP16 (32 lanes) with a=1,b=0" in {
    test(new MapBeatRtDUT)
      .withAnnotations(Seq(VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        val rng = new Random(0x3C)
        dut.io.a.poke(f32bits(1.0f).U); dut.io.b.poke(f32bits(0.0f).U)
        // FP8: 64 bytes, exp != 0x1f (no inf/nan) so widen->FP32->narrow round-trips exactly => out == in
        dut.io.fmt.poke(FpHelpers.FMT_FP8.U)
        for (_ <- 0 until 100) {
          // random NORMAL FP8 values (exp 1..30): no inf/nan (exp!=31), no signed-zero, no subnormals. The
          // narrow unit FTZ's FP8 subnormals (a real FP8 numeric property, matched by narrowRt), and a*x+b
          // maps -0.0 -> +0.0 (correct FP) — both would break a strict identity, so restrict to normals here.
          val bytes = Seq.fill(64) { (if (rng.nextBoolean()) 0x80 else 0) | ((1 + rng.nextInt(30)) << 2) | rng.nextInt(4) }
          val beat  = bytes.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (v, i)) => acc | (BigInt(v & 0xff) << (8 * i)) }
          dut.io.inBeat.poke(beat.U); dut.clock.step(1)
          assert(dut.io.outBeat.peekInt() == beat, "FP8 identity beat mismatch (64-lane round-trip)")
        }
        // FP16: 32 halfwords, identity round-trip
        dut.io.fmt.poke(FpHelpers.FMT_FP16.U)
        for (_ <- 0 until 100) {
          val xs   = Seq.fill(32)(f32ToF16bits(rng.between(-8, 8) + rng.nextInt(4) * 0.25f))
          val beat = xs.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (h, i)) => acc | (BigInt(h & 0xffff) << (16 * i)) }
          dut.io.inBeat.poke(beat.U); dut.clock.step(1)
          assert(dut.io.outBeat.peekInt() == beat, "FP16 identity beat mismatch (32-lane round-trip)")
        }
        println("[RuntimePrecision] full-beat identity round-trip OK: FP8 (64 lanes) + FP16 (32 lanes), one netlist")
      }
  }
}
