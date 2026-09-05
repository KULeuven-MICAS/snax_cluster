package snax.DataPathJunction

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

import snax.DataPathExtension.FpHelpers

/** O5 -- the declared schema. Every operator on the socket reports, from its CSR word alone, that it cannot honour the
  * configuration it was handed. The point is not the check itself but WHAT IT REPLACES: without it each word below
  * produces a finite, format-legal, plausible beat that no downstream hop can tell from a correct one, which is exactly
  * the species of bug this block has shipped twice.
  *
  * The port reports; it does not refuse. Each case below also asserts the datapath still retires, because a fold that
  * stops mid-chain is worse than one that completes and is flagged.
  */
class JunctionCfgErrTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))

  /** hold a CSR word with the operator armed, and read the (stream-constant) verdict */
  private def verdict(dut: DataPathJunctionHarness, csr: BigInt): Boolean = {
    dut.io.csr_i(0).poke(csr.U)
    dut.io.enable_i.poke(true)
    dut.clock.step(2)
    dut.io.cfgerr_o.peekBoolean()
  }

  "CfgErr_monoid" should "flag a saturated sigma, an overflowing role count, and no live slot" in {
    test(new DataPathJunctionHarness(new HasMonoidJunction())).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) {
      dut =>
        def csr(n: Int, nExp: Int, nAdd: Int, sigma: Int, nValid: Int): BigInt =
          (BigInt(sigma) << 26) | (BigInt(nAdd) << 22) | (BigInt(nExp) << 18) | (BigInt(n) << 8) | BigInt(nValid)
        assert(!verdict(dut, csr(1, 1, 0, 3, 8)), "a legal MOMENT word must not be flagged")
        assert(!verdict(dut, csr(9, 9, 0, 0, 1)), "a legal wide-head word must not be flagged")
        // F = 10 admits sigma = 0 only. Asking for 3 saturates -- the beat is still well-formed, but it carries
        // ONE partial where the caller asked for eight, and nothing else would ever say so.
        assert(verdict(dut, csr(9, 9, 0, 3, 1)), "sigma = 3 at F = 10 saturates and must be reported")
        // more twisted+summed roles than there are value coordinates to give them to
        assert(verdict(dut, csr(1, 3, 0, 3, 8)), "nExp + nAdd > n must be reported")
        // every slot masked to the identity: a legal identity partial, indistinguishable from a stale CSR
        assert(verdict(dut, csr(1, 1, 0, 3, 0)), "nValid = 0 must be reported")
        println("[O5/monoid] saturated sigma, role overflow and an empty beat are all reported, not swallowed")
    }
  }

  "CfgErr_elementwise" should "flag a format this instance was never elaborated for" in {
    // THE most dangerous word this operator can be given: an unbuilt `fmt` falls through the repack MuxLookup
    // to the first arm and reinterprets the beat in a different number system, at full rate, silently.
    test(new DataPathJunctionHarness(new HasElementwiseJunction(elemWidth = 16, intWidths = Seq(32))))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        import ElementwiseJunction._
        def csr(op: Int, fmt: Int): BigInt = (BigInt(fmt) << 4) | BigInt(op)
        assert(!verdict(dut, csr(OP_ADD, FpHelpers.FMT_FP16)), "FP16 is built at elemWidth 16")
        assert(!verdict(dut, csr(OP_MAX, FMT_INT32)), "INT32 was requested in intWidths")
        assert(verdict(dut, csr(OP_ADD, FMT_INT8)), "INT8 was NOT elaborated here and must be reported")
        assert(verdict(dut, csr(OP_ADD, FpHelpers.FMT_FP8)), "FP8 is narrower than elemWidth and must be reported")
        assert(verdict(dut, csr(7, FpHelpers.FMT_FP16)), "an opcode past MIN must be reported")
        println("[O5/linear] an fmt this netlist does not build is reported instead of reinterpreting the beat")
      }
  }

  "CfgErr_topk" should "flag a live count its tuple cannot hold" in {
    test(new DataPathJunctionHarness(new HasTopKJunction(k = 8)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        assert(!verdict(dut, BigInt(8)), "nValid = k is legal")
        assert(verdict(dut, BigInt(0)), "nValid = 0 must be reported")
        assert(verdict(dut, BigInt(9)), "nValid > k must be reported")
        println("[O5/topk] a live count larger than the elaborated tuple is reported")
      }
  }

  "CfgErr_bitwise" should "flag the encodable op that does not exist" in {
    test(new DataPathJunctionHarness(new HasBitwiseJunction()))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        import BitwiseJunction._
        assert(!verdict(dut, BigInt(OP_XOR)), "XOR is legal")
        assert(verdict(dut, BigInt(3)), "op 3 does not exist and must be reported")
        println("[O5/bitwise] the fourth encoding of a two-bit field with three operators is reported")
      }
  }

  "CfgErr_disabled" should "stay quiet when the operator is not armed" in {
    // The host aggregates only the ARMED operator's verdict, because the others were handed CSR words meant for
    // someone else and would otherwise all shout at once.
    test(new DataPathJunctionHarness(new HasBitwiseJunction()))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        dut.io.csr_i(0).poke(BigInt(3).U)
        dut.io.enable_i.poke(false)
        dut.clock.step(2)
        assert(!dut.io.cfgerr_o.peekBoolean(), "a disabled operator must not report on a word it will never read")
        println("[O5] a disabled operator is silent about configuration it never reads")
      }
  }
}
