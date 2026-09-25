// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

package snax.DataPathExtension

import chisel3._
import chisel3.util._

/** Int32ColumnScale: y = round_rne(x * f[col]) on a stream of INT32 accumulator beats, one FP16 factor per COLUMN.
  *
  * WHAT IT IS FOR. FlashAttention's online softmax rescales the running output whenever a key tile raises a query's
  * maximum: O <- corr[q] * O + P.V, with one corr per query -- per COLUMN of O^T. The GEMM accumulates O in INT32
  * through its own C input, and nothing between that accumulator and the array can multiply it by a float, so the
  * rescale simply did not happen (correct only while the max never moves). Sitting on the C READ path, this block
  * scales C as it streams into the array, so the matmul that adds P.V already sees corr * O: the recurrence becomes
  * exact at zero extra passes and zero extra traffic.
  *
  * WHICH COLUMN A LANE IS. The C/D port serialises every Mu x Nu output block into `beatsPerBlock` beats of
  * dataWidth/32 INT32 lanes, element e = beat * lanes + lane in row-major (r, c) order, so lane l carries column
  * l % colsPerBlock. Blocks arrive n-inner (the D32 descriptor walks chunks, then N, then M), so the block column is
  * (beat / beatsPerBlock) % N, with N -- the output's column-block count -- set per task in csr(0).
  *
  * CSRs: csr(0)[7:0] = N; csr(1 + i) = {f[2i + 1], f[2i]}, the factors as packed FP16 -- exactly the layout of a
  * 64-byte FP16 beat holding one factor per column, so software copies its 16 words without touching them.
  *
  * ARITHMETIC. f = (-1)^s * M * 2^(e - 25) with M = 1024 + mant (e > 0) or mant (e = 0, then shift 24). The product
  * x * M is exact (43 bits), shifted right with round-to-nearest-even, negated for s, and saturated to INT32. An
  * infinite or NaN factor saturates by sign. For corr in [0, 1] the shift is 10..24, so nothing saturates.
  */
/** One beat between the multiply and the round stages. */
class Int32ColumnScaleProd(lanes: Int) extends Bundle {
  val p   = Vec(lanes, SInt(44.W)) // x * M, exact
  val sh  = Vec(lanes, SInt(7.W))  // right shift; negative = left
  val neg = Vec(lanes, Bool())     // factor sign
  val inf = Vec(lanes, Bool())     // factor is Inf/NaN: saturate by the product's sign
}

class Int32ColumnScale(
  colsPerBlock:  Int = 16,
  beatsPerBlock: Int = 8,
  maxColBlocks:  Int = 2
)(implicit extensionParam: DataPathExtensionParam)
    extends DataPathExtension {

  val lanes   = extensionParam.dataWidth / 32
  val maxCols = colsPerBlock * maxColBlocks
  require(extensionParam.dataWidth % 32 == 0, "Int32ColumnScale: dataWidth must be a multiple of 32")
  require(lanes % colsPerBlock == 0, "Int32ColumnScale: a beat must hold whole rows of a block")
  require(maxCols % 2 == 0, "Int32ColumnScale: factors are packed two per CSR")

  // ---- configuration --------------------------------------------------------------------------------------
  val nBlocks = ext_csr_i(0)(7, 0)
  val factors = VecInit((0 until maxCols).map(c => ext_csr_i(1 + c / 2)(16 * (c % 2) + 15, 16 * (c % 2))))

  // ---- which column block this beat belongs to --------------------------------------------------------------
  val beatInBlock = RegInit(0.U(log2Ceil(beatsPerBlock).max(1).W))
  val colBlock    = RegInit(0.U(8.W))
  val accept      = ext_data_i.fire
  when(ext_start_i) {
    beatInBlock := 0.U; colBlock := 0.U
  }.elsewhen(accept) {
    val lastBeat = beatInBlock === (beatsPerBlock - 1).U
    beatInBlock := Mux(lastBeat, 0.U, beatInBlock + 1.U)
    when(lastBeat) { colBlock := Mux(colBlock + 1.U >= nBlocks, 0.U, colBlock + 1.U) }
  }

  // ---- stage 1: the exact product and the shift it needs ------------------------------------------------------
  val s1   = Module(new Queue(new Int32ColumnScaleProd(lanes), 1, pipe = true))
  val xs   = ext_data_i.bits.asTypeOf(Vec(lanes, SInt(32.W)))
  val prod = Wire(new Int32ColumnScaleProd(lanes))
  for (l <- 0 until lanes) {
    val col  = (colBlock * colsPerBlock.U + (l % colsPerBlock).U)(log2Ceil(maxCols) - 1, 0)
    val f    = factors(col)
    val e    = f(14, 10)
    val mant = f(9, 0)
    val m    = Mux(e === 0.U, Cat(0.U(1.W), mant), Cat(1.U(1.W), mant)) // 11 bits
    prod.p(l)   := xs(l) * Cat(0.U(1.W), m).asSInt                    // 32 x 12 signed -> 44
    prod.sh(l)  := Mux(e === 0.U, 24.S(7.W), 25.S(7.W) - Cat(0.U(2.W), e).asSInt)
    prod.neg(l) := f(15)
    prod.inf(l) := e === 31.U
  }
  s1.io.enq.valid  := ext_data_i.valid && !ext_start_i
  s1.io.enq.bits   := prod
  ext_data_i.ready := s1.io.enq.ready && !ext_start_i

  // ---- stage 2: round, sign, saturate --------------------------------------------------------------------
  val s2   = Module(new Queue(UInt(extensionParam.dataWidth.W), 1, pipe = true))
  val maxI = BigInt(Int.MaxValue)
  val minI = BigInt(Int.MinValue)
  val res  = Wire(Vec(lanes, SInt(32.W)))
  for (l <- 0 until lanes) {
    val p   = s1.io.deq.bits.p(l)
    val sh  = s1.io.deq.bits.sh(l)
    // right shift by sh >= 1 with round-to-nearest-even; floor division keeps the remainder non-negative, so
    // the same rule holds for negative products.
    val shr     = sh.asUInt(4, 0) // 1..24 when sh > 0
    val q       = (p >> shr).asSInt
    val half    = ((p >> (shr - 1.U)).asUInt)(0)
    val stickyM = ((1.U(44.W) << (shr - 1.U)) - 1.U)(43, 0)
    val sticky  = (p.asUInt & stickyM) =/= 0.U
    val up      = half && (sticky || q.asUInt(0))
    val rShift  = q +& up.asUInt.zext
    val lShift  = (p << (-sh).asUInt(2, 0)).asSInt
    val mag     = Mux(sh > 0.S, rShift, Mux(sh === 0.S, p, lShift))
    val signed  = Mux(s1.io.deq.bits.neg(l), -mag, mag)
    val satd    = Mux(signed > maxI.S, maxI.S(32.W), Mux(signed < minI.S, minI.S(32.W), signed(31, 0).asSInt))
    val infSat  = Mux(p < 0.S =/= s1.io.deq.bits.neg(l), minI.S(32.W), maxI.S(32.W))
    res(l) := Mux(s1.io.deq.bits.inf(l), Mux(p === 0.S, 0.S(32.W), infSat), satd)
  }
  s2.io.enq.valid := s1.io.deq.valid
  s2.io.enq.bits  := res.asUInt
  s1.io.deq.ready := s2.io.enq.ready

  ext_data_o.valid := s2.io.deq.valid
  ext_data_o.bits  := s2.io.deq.bits
  s2.io.deq.ready  := ext_data_o.ready

  ext_busy_o := s1.io.count =/= 0.U || s2.io.count =/= 0.U
}

class HasInt32ColumnScale(
  dataWidth:     Int = 1024,
  colsPerBlock:  Int = 16, // meshCol: the columns one output block spans
  beatsPerBlock: Int = 8,  // C/D beats per output block: Mu * Nu * 32 / dataWidth
  maxColBlocks:  Int = 2   // the widest output this rescales, in column blocks (FA: Br / meshCol)
) extends HasDataPathExtension {
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName = "Int32ColumnScale",
      userCsrNum = 1 + colsPerBlock * maxColBlocks / 2, // N, then the factors packed two per CSR
      dataWidth  = dataWidth
    )

  def instantiate(clusterName: String): Int32ColumnScale =
    Module(
      new Int32ColumnScale(colsPerBlock, beatsPerBlock, maxColBlocks) {
        override def desiredName = clusterName + namePostfix
      }
    )
}
