package snax.DataPathJunction

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

import snax.DataPathJunction.JunctionTestUtils._
import snax.utils.TestFp16

/** `SoftmaxStatsJunction` BEHIND THE SOCKET: the host with three operators, the stats junction needing TWO user CSRs
  * and sitting between two that need one each.
  *
  * What the unit's own tester cannot show is that the host hands it the right CSR words. The cfg word list is the
  * concatenation of every operator's CSRs, so the stats junction reads indices 1 and 2, the Elementwise junction index 0
  * and TopK index 3. Every case below loads the CSRs of the operators that are NOT armed with words that would be plainly
  * wrong if read (a zero live count, a NaN scale, an op that does not exist), so a mis-routed CSR changes the answer.
  */
class SoftmaxStatsHostTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))

  private val ROWS = 32
  private def enc(d: Double): Int    = TestFp16.enc(d) & 0xffff
  private def dec(b: Int):    Double = TestFp16.dec(b).toDouble
  private def beat(rows: Seq[Int]): BigInt = rows.zipWithIndex.map { case (x, i) => BigInt(x) << (16 * i) }.sum
  private def rowsOf(b: BigInt): Seq[Int] = (0 until ROWS).map(i => ((b >> (16 * i)) & 0xffff).toInt)

  private def operators: Seq[HasDataPathJunction] =
    Seq(new HasElementwiseJunction(elemWidth = 16, fpPipe = 1), new HasSoftmaxStatsJunction(), new HasTopKJunction(k = 8))
  private val EW = 0; private val ST = 1; private val TK = 2

  private val ewAddFp16   = BigInt(0x00)            // op ADD, fmt FP16
  private val ewBad       = BigInt(0x07)            // an op that does not exist
  private val topkBad     = BigInt(9)               // nValid above k
  private val statsBad    = Seq(BigInt(0), BigInt(0x7fc00000L)) // nValid = 0, a NaN scale

  private def csrs(ew: BigInt, st0: BigInt, st1: BigInt, tk: BigInt): Seq[BigInt] = Seq(ew, st0, st1, tk)

  /** Drive `aBeats` and `bBeats` through the host with the given enable word; a side given as None is never offered. */
  private def run(
    dut:    JunctionHostHarness,
    enable: Long,
    words:  Seq[BigInt],
    aBeats: Option[Seq[BigInt]],
    bBeats: Option[Seq[BigInt]]
  ): Seq[BigInt] = {
    for ((c, i) <- words.zipWithIndex) dut.io.cfg.userCsr(i).poke(c.U)
    dut.io.cfg.enable.poke(enable.U)
    dut.io.start.poke(true); dut.clock.step(1); dut.io.start.poke(false)
    val n   = aBeats.orElse(bBeats).get.length
    val out = scala.collection.mutable.ArrayBuffer[BigInt]()
    def feed(io: chisel3.util.DecoupledIO[UInt], beats: Option[Seq[BigInt]]): Unit = beats match {
      case None => io.valid.poke(false)
      case Some(bs) =>
        for (bt <- bs) {
          io.bits.poke(bt.U); io.valid.poke(true)
          var g = 0
          while (!io.ready.peekBoolean() && g < 400) { dut.clock.step(1); g += 1 }
          assert(g < 400, "an operand beat was never accepted")
          dut.clock.step(1)
        }
        io.valid.poke(false)
    }
    var th = new chiseltest.internal.TesterThreadList(Seq())
    th = th.fork { feed(dut.io.data.a, aBeats) }
    th = th.fork { feed(dut.io.data.b, bBeats) }
    th = th.fork {
      dut.io.data.out.ready.poke(true)
      for (_ <- 0 until n) {
        var g = 0
        while (!dut.io.data.out.valid.peekBoolean() && g < 600) { dut.clock.step(1); g += 1 }
        assert(g < 600, "no output beat retired")
        out += dut.io.data.out.bits.peekInt()
        dut.clock.step(1)
      }
    }
    th.joinAndStep()
    var w = 0; while (dut.io.busy.peekBoolean() && w < 400) { dut.clock.step(1); w += 1 }
    assert(!dut.io.busy.peekBoolean(), "the host must go idle once the group has left")
    out.toSeq
  }

  private def group(rng: Random): (Seq[Int], Seq[Int]) =
    (Seq.fill(ROWS)(enc(rng.nextGaussian() * 3.0)), Seq.fill(ROWS)(enc(rng.between(1.0, 600.0))))

  "StatsHost_routing" should "give the stats junction CSRs 1 and 2, and leave its neighbours their own" in {
    test(new JunctionHostHarness(operators)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng = new Random(0x5714)

      // ---- arm the stats junction; the Elementwise and TopK words are wrong on purpose ------------------------
      val (ka, la) = group(rng); val (kb, lb) = group(rng)
      val o = run(dut, 1L << ST, csrs(ewBad, BigInt(ROWS), BigInt(java.lang.Float.floatToIntBits(0.5f).toLong), topkBad),
        Some(Seq(beat(ka), beat(la))), Some(Seq(beat(kb), beat(lb))))
      var worst = 0.0
      for (r <- 0 until ROWS) {
        val (a, b) = (dec(ka(r)), dec(kb(r))); val m = math.max(a, b)
        val want   = math.exp(0.5 * (a - m)) * dec(la(r)) + math.exp(0.5 * (b - m)) * dec(lb(r))
        assert(rowsOf(o(0))(r) == (if (a >= b) ka(r) else kb(r)), s"key row $r")
        worst = math.max(worst, math.abs(dec(rowsOf(o(1))(r)) - want) / want)
      }
      assert(worst <= 6e-4, f"worst relative error $worst%.2e: the stats junction did not read its own CSRs")
      println(f"[StatsHost] stats armed behind the host: worst error $worst%.2e of the golden with scale 0.5 from CSR index 2")

      // ---- arm the Elementwise junction; the stats CSRs are wrong on purpose ----------------------------------
      val a = Seq.tabulate(ROWS)(i => enc((i + 1).toDouble)); val b = Seq.tabulate(ROWS)(i => enc(2.0 * (i % 5 + 1)))
      val e = run(dut, 1L << EW, csrs(ewAddFp16, statsBad(0), statsBad(1), topkBad), Some(Seq(beat(a))), Some(Seq(beat(b))))
      for (r <- 0 until ROWS)
        assert(dec(rowsOf(e(0))(r)) == dec(a(r)) + dec(b(r)), s"elementwise lane $r read the wrong CSR")
      println("[StatsHost] elementwise armed behind the host: unaffected by the stats junction's CSRs")
    }
  }

  "StatsHost_cfgerr" should "report only the armed operator's verdict" in {
    test(new JunctionHostHarness(operators)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      def verdict(en: Long, w: Seq[BigInt]): Boolean = {
        for ((c, i) <- w.zipWithIndex) dut.io.cfg.userCsr(i).poke(c.U)
        dut.io.cfg.enable.poke(en.U); dut.clock.step(2)
        dut.io.cfgerr.peekBoolean()
      }
      assert(verdict(1L << ST, csrs(ewBad, statsBad(0), BigInt(0), topkBad)), "an empty live count must be reported")
      assert(verdict(1L << ST, csrs(ewBad, BigInt(8), statsBad(1), topkBad)), "a NaN scale must be reported")
      assert(!verdict(1L << ST, csrs(ewBad, BigInt(8), BigInt(0), topkBad)), "a legal stats word must not be reported")
      assert(!verdict(1L << EW, csrs(ewAddFp16, statsBad(0), statsBad(1), topkBad)), "a bad stats word must not shout when it is not armed")
      println("[StatsHost] cfgerr follows the armed operator, and only it")
    }
  }

  "StatsHost_substitution" should "stand in for a shard that never arrives, with a two-beat group" in {
    test(new JunctionHostHarness(operators)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng      = new Random(0x5ab5)
      val (kb, lb) = group(rng)
      val live     = Seq(beat(kb), beat(lb))
      val words    = csrs(ewBad, BigInt(ROWS), BigInt(0), topkBad)
      val onB      = run(dut, (1L << ST) | (1L << 16), words, None, Some(live))   // A never arrives
      assert(onB == live, "substituting A did not reproduce the surviving operand")
      val onA      = run(dut, (1L << ST) | (1L << 17), words, Some(live), None)   // B never arrives
      assert(onA == live, "substituting B did not reproduce the surviving operand")
      println("[StatsHost] a dead shard is a configuration: the identity group stands in and the live group comes out unchanged")
    }
  }
}
