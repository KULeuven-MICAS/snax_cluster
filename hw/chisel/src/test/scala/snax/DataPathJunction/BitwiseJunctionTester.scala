package snax.DataPathJunction

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

import snax.DataPathJunction.JunctionTestUtils._

/** Tier-1 for `BitwiseJunction` -- the operator with no arithmetic.
  *
  * The tests are the CONTRACT tests, because the arithmetic is one gate and needs no defending. What is being checked
  * is that an operator containing no adder, no multiplier, no comparator and no lane structure still satisfies the same
  * four obligations the twisted family does.
  */
class BitwiseJunctionTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))

  import BitwiseJunction._

  private val ALL_ONES = (BigInt(1) << 512) - 1

  private def hasBw = new HasBitwiseJunction()
  private def csr(op:   Int):    BigInt = BigInt(op)
  private def rand(rng: Random): BigInt = BigInt(512, rng)

  private val ops = Seq(
    (OP_OR, "OR", (a: BigInt, b: BigInt) => a | b, BigInt(0)),
    (OP_AND, "AND", (a: BigInt, b: BigInt) => a & b, ALL_ONES),
    (OP_XOR, "XOR", (a: BigInt, b: BigInt) => a ^ b, BigInt(0))
  )

  "Bitwise_ops" should "reduce two beats with OR / AND / XOR" in {
    test(new DataPathJunctionHarness(hasBw)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng = new Random(0xb17)
      for ((op, name, ref, _) <- ops) {
        for (trial <- 0 until 4) {
          val (a, b) = (rand(rng), rand(rng))
          val out    = runPair(dut, csr(op), a, b, bDelay = trial % 3)
          assert(out == ref(a, b), f"$name trial $trial: 0x$out%x != 0x${ref(a, b)}%x")
        }
      }
      println("[Bitwise] OR / AND / XOR exact over 512 bits, 4 trials each")
    }
  }

  "Bitwise_O3_identity" should "leave a beat untouched when folded against its op's identity" in {
    // O3 without any lane structure at all. Every other operator on this socket pads PER SLOT; this one's
    // identity is a whole-beat constant, and which constant it is depends on the op -- all ones for AND, all
    // zeros for OR and XOR. A chassis that synthesised a missing operand as zeros would silently annihilate
    // every AND fold in the chain.
    test(new DataPathJunctionHarness(hasBw)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng = new Random(0x1de)
      for ((op, name, _, id) <- ops) {
        val a = rand(rng)
        assert(runPair(dut, csr(op), a, id) == a, s"$name: folding against its identity changed the beat")
        assert(runPair(dut, csr(op), id, a) == a, s"$name: identity is not two-sided")
      }
      println("[Bitwise/O3] identity is a whole-beat constant per op: 0 for OR/XOR, all-ones for AND")
    }
  }

  "Bitwise_O2_closure" should "let its own output re-enter as an input" in {
    test(new DataPathJunctionHarness(hasBw)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng = new Random(0xc105)
      for ((op, name, ref, _) <- ops) {
        val (a, b, c) = (rand(rng), rand(rng), rand(rng))
        val ab        = runPair(dut, csr(op), a, b)
        val abc       = runPair(dut, csr(op), ab, c, bDelay = 2)
        assert(abc == ref(ref(a, b), c), s"$name: the output beat is not a legal input beat")
      }
      println("[Bitwise/O2] a folded beat re-enters unchanged and a third shard folds in")
    }
  }

  "Bitwise_idempotence" should "distinguish the duplicate-tolerant ops from XOR" in {
    // The reason duplicate tolerance has to be declared PER OP rather than per operator. OR and AND are
    // idempotent, so a retransmitted beat is a no-op and the chain survives it. XOR is self-inverse, so the
    // same retransmission cancels the shard entirely -- a wrong answer that is perfectly well-formed.
    test(new DataPathJunctionHarness(hasBw)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng    = new Random(0x1d3)
      val (a, b) = (rand(rng), rand(rng))
      for ((op, name, ref, _) <- ops) {
        val once  = runPair(dut, csr(op), a, b)
        val twice = runPair(dut, csr(op), once, b)
        if (op == OP_XOR) assert(twice == a, "XOR should be self-inverse: folding b twice must cancel it")
        else assert(twice == once, s"$name should be idempotent: folding b twice must be a no-op")
      }
      println("[Bitwise] OR/AND absorb a duplicated beat; XOR cancels the shard -- declare it per OP")
    }
  }

  "Bitwise_bloom" should "union four Bloom filters along a chain" in {
    // The use case, and the reason OR is worth a gate: a Bloom filter's union IS a bitwise OR, so merging the
    // set-membership summaries of four shards is one chained fold with no arithmetic anywhere in the path.
    test(new DataPathJunctionHarness(hasBw)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng   = new Random(0xb100)
      // each shard sets ~24 bits, as a small filter would
      val shard = Seq.fill(4)((0 until 24).map(_ => BigInt(1) << rng.between(0, 512)).reduce(_ | _))
      var acc   = shard(0)
      for (h <- 1 until 4) acc = runPair(dut, csr(OP_OR), acc, shard(h), bDelay = h)
      assert(acc == shard.reduce(_ | _), "the chained union is not the OR of the four filters")
      assert(runPair(dut, csr(OP_OR), acc, shard(2)) == acc, "re-folding a shard changed the union")
      println("[Bitwise/bloom] 4 filters unioned over a chain; re-folding a shard is a no-op")
    }
  }

  "Bitwise_O1_O4" should "retire at a fixed latency and hold no state between pairs" in {
    test(new DataPathJunctionHarness(hasBw)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng   = new Random(0x0141)
      val pairs = Seq.fill(6)((rand(rng), rand(rng)))
      // O4: the same pairs, run in reverse order with a varying operand skew, must give the same results
      val fwd   = pairs.map { case (a, b) => runPair(dut, csr(OP_XOR), a, b) }
      val rev   = pairs.indices.reverse.map { i =>
        i -> runPair(dut, csr(OP_XOR), pairs(i)._1, pairs(i)._2, bDelay = i % 4)
      }.toMap
      for (i <- pairs.indices) assert(rev(i) == fwd(i), s"pair $i differs when interleaved")

      // O1: first output at the same cycle whether the payload is 1 beat or 64
      def firstOut(n: Int): (Int, Double) = {
        val beats = Seq.fill(n)(rand(rng))
        dut.io.csr_i(0).poke(csr(OP_OR).U)
        dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
        dut.io.out_o.ready.poke(true)
        dut.io.a_i.valid.poke(true); dut.io.b_i.valid.poke(true)
        dut.io.a_i.bits.poke(beats(0).U); dut.io.b_i.bits.poke(beats(0).U)
        var cyc   = 0; var fed = 0; var got = 0; var first = -1; var warm = -1; var last = -1
        while (got < n && cyc < n * 40 + 600) {
          val canFeed = fed < n && dut.io.a_i.ready.peekBoolean() && dut.io.b_i.ready.peekBoolean()
          val outNow  = dut.io.out_o.valid.peekBoolean()
          dut.clock.step(1); cyc += 1
          if (canFeed) {
            fed += 1
            if (fed == 4) warm = cyc
            last               = cyc
            if (fed < n) { dut.io.a_i.bits.poke(beats(fed).U); dut.io.b_i.bits.poke(beats(fed).U) }
            else { dut.io.a_i.valid.poke(false); dut.io.b_i.valid.poke(false) }
          }
          if (outNow) { if (first < 0) first = cyc; got += 1 }
        }
        dut.io.out_o.ready.poke(false)
        assert(got == n, s"only $got/$n retired")
        (first, if (last > warm) (n - 4).toDouble / (last - warm) else 0.0)
      }
      val (l1, _) = firstOut(1)
      val (l64, u) = firstOut(64)
      assert(l64 <= l1 + 1, s"NOT cut-through: first output moved from $l1 to $l64 as the payload grew")
      assert(u > 0.9, f"a one-gate fold should stream at one pair per cycle, got $u%.3f")
      println(f"[Bitwise/O1+O4] first-out N=1 -> $l1 CC, N=64 -> $l64 CC; util=$u%.3f; stateless across pairs")
    }
  }
}
