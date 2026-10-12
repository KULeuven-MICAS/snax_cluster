package snax.simd

import scala.util.Random

import chisel3._

import chiseltest._
import chiseltest.simulator.VerilatorFlags
import snax.DataPathExtension.DataPathExtensionHarness

/** Bit-level FP helpers that mirror the SIMD datapath's rounding exactly. */
object SimdFp {
  def f16ToF32(h: Int): Float = {
    val sign = if ((h & 0x8000) != 0) -1.0 else 1.0
    val exp  = (h >> 10) & 0x1f
    val mant = h & 0x3ff
    val v: Double =
      if (exp == 0) sign * mant * math.pow(2, -24)
      else if (exp == 0x1f) { if (mant == 0) sign * Double.PositiveInfinity else Double.NaN }
      else sign * (1024 + mant) * math.pow(2, exp - 25)
    v.toFloat
  }

  /** FP32 -> FP16, round to nearest even, subnormals kept, overflow to Inf, NaN to the canonical 0x7E00. */
  def f32ToF16(f: Float): Int = {
    val bits = java.lang.Float.floatToRawIntBits(f)
    val sign = (bits >>> 16) & 0x8000
    val rawe = (bits >>> 23) & 0xff
    val mant = bits & 0x7fffff
    if (rawe == 0xff) return if (mant != 0) 0x7e00 else sign | 0x7c00
    val exp  = rawe - 127 + 15
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
    var h = sign | (exp << 10) | (mant >>> 13)
    val rem = mant & 0x1fff
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1)) h += 1
    h
  }

  def bits32(f: Float): Long = java.lang.Float.floatToRawIntBits(f).toLong & 0xffffffffL

  /** FpHelpers.fp32max, bit for bit: a sign-magnitude compare that returns `a` on a tie. */
  def max32(a: Float, b: Float): Float = {
    val ab    = bits32(a); val bb       = bits32(b)
    val sa    = (ab >> 31) == 1; val sb = (bb >> 31) == 1
    val aWins = if (sa != sb) !sa else if (sa) bb >= ab else ab >= bb
    if (aWins) a else b
  }

  /** A balanced pairwise tree, odd element out carried up a level -- SimdStreamOp.treeReduce. */
  def tree[T](xs: Seq[T])(node: (T, T) => T): T =
    if (xs.length == 1) xs.head
    else tree(xs.grouped(2).map { case Seq(a, b) => node(a, b); case Seq(a) => a }.toSeq)(node)

  /** A random FP16 value on a coarse grid, so sums of many of them stay in range. */
  def randF16(rnd: Random, mag: Int): Int = f32ToF16(rnd.between(-mag, mag) + rnd.nextInt(8) * 0.125f)

  def pack(lanes: Seq[Int], w: Int): BigInt = {
    val mask = (BigInt(1) << w) - 1
    lanes.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (v, i)) => acc | ((BigInt(v) & mask) << (w * i)) }
  }

  def unpack(beat: BigInt, w: Int, n: Int): Seq[Int] = {
    val mask = (BigInt(1) << w) - 1
    (0 until n).map(i => ((beat >> (w * i)) & mask).toInt)
  }
}

/** One task on an operator: its CSRs, its input beats and how many output beats it must produce. */
case class SimdTask(csr: Seq[BigInt], in: Seq[BigInt], nOut: Int, name: String = "")

/** Drives a `DataPathExtensionHarness` through a sequence of tasks, as SimdTop would: CSRs, a start pulse, the stream,
  * then wait for busy to fall before the next task. `inGap` / `outStall` are the probabilities of an idle input cycle
  * and of a not-ready output cycle, so the credit and FIFO logic sees back-pressure in every phase. Returns each task's
  * output beats and the number of cycles its input was held off (valid high, ready low).
  */
object SimdStreamDriver {
  val annotations = Seq(VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))

  def run(
    dut:      DataPathExtensionHarness,
    tasks:    Seq[SimdTask],
    seed:     Int,
    inGap:    Double,
    outStall: Double
  ): Seq[(Seq[BigInt], Int)] = {
    val rnd   = new Random(seed)
    val limit = 200000
    dut.io.enable_i.poke(true.B)
    dut.io.data_i.valid.poke(false.B)
    dut.io.data_o.ready.poke(false.B)
    tasks.map { t =>
      t.csr.zipWithIndex.foreach { case (v, i) => dut.io.csr_i(i).poke(v.U) }
      dut.io.start_i.poke(true.B); dut.clock.step(1); dut.io.start_i.poke(false.B)

      val outs    = scala.collection.mutable.ArrayBuffer[BigInt]()
      var held    = 0
      var threads = new chiseltest.internal.TesterThreadList(Seq())
      val inSeed  = rnd.nextInt(); val outSeed = rnd.nextInt()
      threads = threads.fork {
        val r = new Random(inSeed)
        for (b <- t.in) {
          while (r.nextDouble() < inGap) { dut.io.data_i.valid.poke(false.B); dut.clock.step(1) }
          dut.io.data_i.valid.poke(true.B)
          dut.io.data_i.bits.poke(b.U)
          var c = 0
          while (!dut.io.data_i.ready.peekBoolean()) {
            dut.clock.step(1); held += 1; c += 1; assert(c < limit, s"${t.name}: input hung")
          }
          dut.clock.step(1)
        }
        dut.io.data_i.valid.poke(false.B)
      }
      threads = threads.fork {
        val r = new Random(outSeed)
        var c = 0
        while (outs.length < t.nOut) {
          val ready = r.nextDouble() >= outStall
          dut.io.data_o.ready.poke(ready.B)
          if (ready && dut.io.data_o.valid.peekBoolean()) outs += dut.io.data_o.bits.peekInt()
          dut.clock.step(1); c += 1
          assert(c < limit, s"${t.name}: only ${outs.length} of ${t.nOut} output beats arrived")
        }
        dut.io.data_o.ready.poke(false.B)
      }
      threads.joinAndStep()

      // Nothing more may come out, and the operator must report done.
      dut.io.data_o.ready.poke(true.B)
      var w = 0
      while (dut.io.busy_o.peekBoolean() && w < 200) {
        assert(!dut.io.data_o.valid.peekBoolean(), s"${t.name}: an extra output beat")
        dut.clock.step(1); w += 1
      }
      assert(!dut.io.busy_o.peekBoolean(), s"${t.name}: busy stuck high after the task")
      for (_ <- 0 until 8) {
        assert(!dut.io.data_o.valid.peekBoolean(), s"${t.name}: an extra output beat")
        dut.clock.step(1)
      }
      dut.io.data_o.ready.poke(false.B)
      (outs.toSeq, held)
    }
  }
}
