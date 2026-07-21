package snax.DataPathExtension

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

/** StreamAttnMergeRt: the COMPLETE distributed flash-attention collective — the (m, ℓ, O) triple. Each beat
  * carries one shard's partial (m, ℓ, O[dHead]); accEn folds them into the global (m*, ℓ*, O*) via the
  * flash-attention merge (m* = max, ell and O rescaled by exp(m_k − m*)). Proves the fold matches the
  * double-precision reference within LUT-exp tolerance, at P=2 and P=4, and that O is rescaled correctly.
  */
class StreamAttnMergeRtTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))
  private val dHead = 8

  private def f32(f: Double): BigInt = BigInt(java.lang.Float.floatToIntBits(f.toFloat).toLong & 0xffffffffL)
  private def dec(b: BigInt): Double = java.lang.Float.intBitsToFloat(b.toInt).toDouble

  // pack (m, ℓ, O[dHead]) into a beat: lane0=m, lane1=ℓ, lanes 2..2+dHead-1 = O
  private def pack(m: Double, l: Double, o: Seq[Double]): BigInt = {
    var b = BigInt(0)
    b |= f32(m) << (32 * 0)
    b |= f32(l) << (32 * 1)
    for ((ov, k) <- o.zipWithIndex) b |= f32(ov) << (32 * (2 + k))
    b
  }
  private def unpack(beat: BigInt): (Double, Double, Seq[Double]) = {
    def lane(i: Int) = dec((beat >> (32 * i)) & ((BigInt(1) << 32) - 1))
    (lane(0), lane(1), (0 until dHead).map(k => lane(2 + k)))
  }

  // double-precision reference flash-attention merge over P shards
  private def refMerge(shards: Seq[(Double, Double, Seq[Double])]): (Double, Double, Seq[Double]) = {
    val mstar = shards.map(_._1).max
    val lstar = shards.map { case (m, l, _) => l * math.exp(m - mstar) }.sum
    val ostar = (0 until dHead).map(j => shards.map { case (m, _, o) => o(j) * math.exp(m - mstar) }.sum)
    (mstar, lstar, ostar)
  }

  // push one beat with CSR bits; slots survive start_i so arm+fold across calls works
  private def push(dut: DataPathExtensionHarness, m: Double, l: Double, o: Seq[Double],
                   accInit: Int, accSlot: Int = 0): (Double, Double, Seq[Double]) = {
    val csr = (BigInt(accSlot) << 10) | (BigInt(accInit) << 9) | (BigInt(1) << 8) | BigInt(1) // accEn=1, nValid=1
    dut.io.csr_i(0).poke(csr.U)
    dut.io.enable_i.poke(true)
    dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
    var out = BigInt(0)
    var th  = new chiseltest.internal.TesterThreadList(Seq())
    th = th.fork {
      dut.io.data_i.bits.poke(pack(m, l, o).U); dut.io.data_i.valid.poke(true)
      while (!dut.io.data_i.ready.peekBoolean()) dut.clock.step(1)
      dut.clock.step(1); dut.io.data_i.valid.poke(false)
    }
    th = th.fork {
      while (!dut.io.data_o.valid.peekBoolean()) dut.clock.step(1)
      out = dut.io.data_o.bits.peekInt()
      dut.io.data_o.ready.poke(true); dut.clock.step(1); dut.io.data_o.ready.poke(false)
    }
    th.joinAndStep()
    dut.io.data_o.ready.poke(true)
    var w = 0; while (dut.io.busy_o.peekBoolean() && w < 200) { dut.clock.step(1); w += 1 }
    dut.io.data_o.ready.poke(false)
    unpack(out)
  }

  private def checkClose(got: (Double, Double, Seq[Double]), gold: (Double, Double, Seq[Double]), msg: String): Unit = {
    assert(math.abs(got._1 - gold._1) <= math.abs(gold._1) * 1e-6 + 1e-9, s"$msg m=${got._1} vs ${gold._1}")
    assert(math.abs(got._2 - gold._2) / gold._2 <= 1.5e-2, s"$msg ℓ=${got._2} vs ${gold._2}")
    for (j <- 0 until dHead) {
      val d = math.abs(got._3(j) - gold._3(j)) / (math.abs(gold._3(j)) + 1e-6)
      assert(d <= 2e-2, s"$msg O[$j]=${got._3(j)} vs ${gold._3(j)} (rel $d)")
    }
  }

  "StreamAttnMergeRt_p2" should "fold two shards' (m,ℓ,O) into the global flash-attention triple" in {
    test(new DataPathExtensionHarness(new HasStreamAttnMergeRt(dHead = dHead)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng = new Random(0x0a11)
        for (trial <- 0 until 10) {
          val a = (rng.between(-2.0, 4.0), rng.between(1.0, 5.0), Seq.fill(dHead)(rng.between(-3.0, 3.0)))
          val b = (rng.between(-2.0, 4.0), rng.between(1.0, 5.0), Seq.fill(dHead)(rng.between(-3.0, 3.0)))
          push(dut, a._1, a._2, a._3, accInit = 1, accSlot = trial % 8) // arm with shard A
          val got = push(dut, b._1, b._2, b._3, accInit = 0, accSlot = trial % 8) // fold shard B
          checkClose(got, refMerge(Seq(a, b)), s"p2 trial $trial")
        }
        println(f"[AttnMergeRt] P=2 (m,ℓ,O) triple fold matches flash-attention reference (dHead=$dHead) over 10 trials")
      }
  }

  "StreamAttnMergeRt_p4" should "fold four shards into one slot (accumulate-on-arrival)" in {
    test(new DataPathExtensionHarness(new HasStreamAttnMergeRt(dHead = dHead)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng = new Random(0x0a44)
        for (trial <- 0 until 6) {
          val shards = Seq.fill(4)((rng.between(-2.0, 5.0), rng.between(1.0, 5.0), Seq.fill(dHead)(rng.between(-3.0, 3.0))))
          push(dut, shards(0)._1, shards(0)._2, shards(0)._3, accInit = 1)
          var got = (0.0, 0.0, Seq.fill(dHead)(0.0))
          for (i <- 1 until 4) got = push(dut, shards(i)._1, shards(i)._2, shards(i)._3, accInit = 0)
          checkClose(got, refMerge(shards), s"p4 trial $trial")
        }
        println(f"[AttnMergeRt] P=4 accEn fold (4 shards -> one slot) matches reference (dHead=$dHead) over 6 trials")
      }
  }
}
