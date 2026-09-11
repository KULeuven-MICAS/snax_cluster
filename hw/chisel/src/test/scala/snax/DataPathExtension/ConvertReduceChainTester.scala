package snax.DataPathExtension

import scala.util.Random

import chisel3._
import chisel3.util._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

import snax.utils._

/** Two extensions wired back to back, which is what an armed multi-extension task builds.
  *
  * FlashAttention fuses Int32ToFp16Converter (extension 0) into StreamReduce (extension 3) so the GEMM's
  * INT32 output is converted and reduced in ONE pass: the reduce runs in tap mode, passing the converted
  * tile through and appending its per-lane statistics. That fused task hangs in the cluster while each
  * operator passes its own tests, so the question this harness answers is whether the pair composes --
  * specifically whether the beat COUNT out is what the writer was programmed for.
  */
class ConvertReduceChain(computeLanes: Int) extends Module with RequireAsyncReset {
  val conv = (new HasInt32ToFp16Converter(dataWidth = 512)).instantiate("chain_conv")
  val red  = (new HasStreamReduce(computeLanes = computeLanes, op = Seq("FMA_FP16", "MAX_FP16"),
                                  elementWidth = 16)).instantiate("chain_red")
  val io = IO(new Bundle {
    val data_i   = Flipped(Decoupled(UInt(512.W)))
    val data_o   = Decoupled(UInt(512.W))
    val conv_csr = Input(Vec(conv.io.csr_i.length, UInt(32.W)))
    val red_csr  = Input(Vec(red.io.csr_i.length, UInt(32.W)))
    val start_i  = Input(Bool())
    val busy_o   = Output(Bool())
  })
  conv.io.csr_i := io.conv_csr
  red.io.csr_i  := io.red_csr
  conv.io.enable_i := true.B
  red.io.enable_i  := true.B
  conv.io.start_i  := io.start_i
  red.io.start_i   := io.start_i
  io.busy_o := conv.io.busy_o || red.io.busy_o

  conv.io.data_i <> io.data_i
  red.io.data_i  <> conv.io.data_o
  io.data_o      <> red.io.data_o
}

class ConvertReduceChainTester extends AnyFlatSpec with ChiselScalatestTester {
  val lanes = 32

  def f16bitsToF32(h: Int): Float = {
    val sign = if ((h & 0x8000) != 0) -1.0 else 1.0
    val exp  = (h >> 10) & 0x1f
    val mant = h & 0x3ff
    val v: Double =
      if (exp == 0) sign * mant * math.pow(2, -24)
      else if (exp == 0x1f) if (mant == 0) sign * Double.PositiveInfinity else Double.NaN
      else sign * (1024 + mant) * math.pow(2, exp - 25)
    v.toFloat
  }

  behavior of "Int32ToFp16 -> StreamReduce chain"

  it should "emit exactly nBeats/2 passthroughs plus ONE statistics beat" in {
    val nIn  = 128            // INT32 beats, as FlashAttention feeds it
    val nFp  = nIn / 2        // converted beats
    test(new ConvertReduceChain(computeLanes = 8))
      .withAnnotations(Seq(VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        dut.clock.setTimeout(0)
        dut.io.conv_csr(0).poke(0.U)
        dut.io.red_csr(0).poke(nFp.U)                 // operandCount = the whole tile
        dut.io.red_csr(1).poke(0x500.U)               // MAX | lanewise bit[10] | tap bit[8]
        dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)

        val rnd = new Random(0xCC)
        val ints = Seq.fill(nIn)(Seq.fill(16)(rnd.between(-3000, 3000)))
        var produced = 0
        var threads = new chiseltest.internal.TesterThreadList(Seq())
        threads = threads.fork {
          dut.io.data_i.valid.poke(true)
          for (b <- ints) {
            val beat = b.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (v, i)) =>
              acc | ((BigInt(v) & ((BigInt(1) << 32) - 1)) << (32 * i))
            }
            var guard = 0
            while (!dut.io.data_i.ready.peekBoolean() && guard < 5000) { dut.clock.step(1); guard += 1 }
            assert(guard < 5000, s"input stalled forever after $produced output beats")
            dut.io.data_i.bits.poke(beat); dut.clock.step(1)
          }
          dut.io.data_i.valid.poke(false)
        }
        threads = threads.fork {
          var idle = 0
          while (idle < 400) {
            if (dut.io.data_o.valid.peekBoolean()) {
              dut.io.data_o.ready.poke(true); dut.clock.step(1); dut.io.data_o.ready.poke(false)
              produced += 1; idle = 0
            } else { dut.clock.step(1); idle += 1 }
          }
        }
        threads.joinAndStep()
        assert(produced == nFp + 1,
               s"expected ${nFp + 1} output beats (${nFp} passthrough + 1 stat), got $produced")
      }
  }
}
