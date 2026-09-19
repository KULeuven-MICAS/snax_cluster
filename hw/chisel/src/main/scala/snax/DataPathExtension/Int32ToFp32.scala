// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

package snax.DataPathExtension

import chisel3._
import chisel3.util._

/** \============================================================================================================
  * `Int32ToFp32` -- the INT32 accumulator's way onto a floating-point collective.
  * \============================================================================================================
  *
  * A GEMM that accumulates in INT32 cannot hand its result to `MonoidJunction`: the fold's twist is
  * `exp(m_lose - m*)`, so every operand it touches is floating point. This extension is the join, and it is the
  * SIMPLEST converter in the tree because of WHERE it sits in the number line rather than any cleverness:
  *
  *   - It CANNOT OVERFLOW. The largest INT32 is 2^31, whose FP32 exponent is 158 against a 255 ceiling. There is
  *     no saturation path and no Inf case to get wrong -- unlike `Int32ToFp16PE`, which saturates to +-Inf and
  *     will do so for any flash-attention numerator worth folding (`sum P8.V8` reaches ~127.l.127, past FP16's
  *     65504 for l > 4).
  *   - It is EXACT for |v| < 2^24, which covers that same numerator with room to spare. Guard, round and sticky
  *     are all zero there, so the rounding logic below is dead weight on the workload that motivated the block
  *     and is present only so the converter is correct for the whole INT32 range.
  *   - It is WIDTH-PRESERVING, 32 bits in and 32 out. So there is no beat repack, no packing counter and no
  *     `extra_loop` CSR -- the three things that make `Int32ToFp16Converter` a state machine rather than a map.
  *
  * Those three properties are the argument for carrying a flash-attention numerator across the fabric in FP32
  * rather than FP16. The narrower transport halves the payload, but it needs the partial re-embedded (a key of
  * `m + ln l`, normalised values, a constant field) purely to fit the range. In FP32 the partial is the one the
  * algebra already writes down: key `m`, values the raw `l` and `O`.
  */
class Int32ToFp32PE(pipelined: Boolean = false) extends Module with RequireAsyncReset {
  val io = IO(new Bundle {
    val in  = Input(SInt(32.W))
    val out = Output(UInt(32.W)) // IEEE-754 binary32
  })

  private def sr[T <: Data](u: T): T = if (pipelined) RegNext(u) else u

  val expBias = 127.U(8.W)
  val SIG     = 23 // FP32 fraction bits

  val sign = io.in(31)
  // `-in` on the SInt then .asUInt: correct for -2^31 too, whose negation is itself in 32 bits and whose
  // unsigned reading is exactly 2^31 -- the magnitude we want.
  val abs  = Mux(io.in < 0.S, (-io.in).asUInt, io.in.asUInt)

  val isZero = abs === 0.U

  // ---- stage 0: normalise. MSB of `abs` to bit 31, and remember which bit it was. ----
  val msbIndex = 31.U - PriorityEncoder(Reverse(abs)) // 0..31
  val magNorm  = (abs << (31.U - msbIndex))(31, 0)

  val sign_d   = sr(sign)
  val zero_d   = sr(isZero)
  val exp_d    = sr(msbIndex)
  val mag_d    = sr(magNorm)

  // ---- stage 1: round to nearest, ties to even, then assemble ----
  // magNorm is 1.xxx in Q1.31, so the fraction is bits 30..8 and the rounding bits are 7 / 6 / 5..0.
  val frac   = mag_d(30, 31 - SIG)
  val guard  = mag_d(31 - SIG - 1)
  val round  = mag_d(31 - SIG - 2)
  val sticky = mag_d(31 - SIG - 3, 0).orR

  val increment = guard && (round || sticky || frac(0)) // RNE
  val fracPlus  = Cat(0.U(1.W), frac) +& increment
  val mantCarry = fracPlus(SIG)
  val fracFinal = Mux(mantCarry, 0.U(SIG.W), fracPlus(SIG - 1, 0))

  // A mantissa carry bumps the exponent. It cannot reach 255: msbIndex <= 31, so exp <= 31 + 127 + 1 = 159.
  val expFinal = (exp_d +& expBias + mantCarry)(7, 0)

  io.out := Mux(zero_d, Cat(sign_d, 0.U(31.W)), Cat(sign_d, expFinal, fracFinal))
}

/** Elaboration wrapper. `numPipe = 0` is a pure combinational map (one register stage still falls out of the
  * output queue); `numPipe = 1` cuts the PE between the normalise and the round, which is where the logic depth
  * actually is.
  */
class HasInt32ToFp32(
  dataWidth: Int = 512,
  numPipe:   Int = 0
) extends HasDataPathExtension {
  require(dataWidth % 32 == 0, s"HasInt32ToFp32: dataWidth ($dataWidth) must be a multiple of 32")
  require(numPipe == 0 || numPipe == 1, s"HasInt32ToFp32: numPipe must be 0 or 1 (got $numPipe)")

  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName = "Int32ToFp32", // -> READER_EXT_INT32TOFP32. Never width-encode the name.
      userCsrNum = 1,             // the socket's mandatory bypass word; this operator has no configuration
      dataWidth  = dataWidth
    )

  def instantiate(clusterName: String): Int32ToFp32 =
    Module(new Int32ToFp32(dataWidth, numPipe) {
      override def desiredName = clusterName + namePostfix
    })
}

class Int32ToFp32(
  dataWidth:      Int = 512,
  numPipe:        Int = 0
)(implicit
  extensionParam: DataPathExtensionParam
) extends DataPathExtension {

  val lanes = dataWidth / 32

  // One PE per lane, by an identity map. No lane ever reads its neighbour, so there is nothing to schedule and
  // no reason to time-multiplex: the converter is cheap enough that `computeLanes` would buy area we are not
  // short of and cost a beat FSM we would then have to test.
  val converted = Cat((0 until lanes).map { i =>
    val pe = Module(new Int32ToFp32PE(pipelined = numPipe > 0))
    pe.io.in := ext_data_i.bits(32 * i + 31, 32 * i).asSInt
    pe.io.out
  }.reverse)

  // Fixed-latency 1:1 map behind a credit, the same shape the junctions use: a beat is only accepted when its
  // result is already guaranteed a slot, so the PE pipeline can never be back-pressured mid-flight.
  val Qdepth = 2 + numPipe
  val outQ   = Module(new Queue(UInt(dataWidth.W), entries = Qdepth))
  val credit = RegInit(Qdepth.U(log2Ceil(Qdepth + 1).W))

  val fire = ext_data_i.valid && credit =/= 0.U
  ext_data_i.ready := credit =/= 0.U

  outQ.io.enq.valid := ShiftRegister(fire, numPipe, false.B, true.B)
  outQ.io.enq.bits  := converted
  assert(!outQ.io.enq.valid || outQ.io.enq.ready, "Int32ToFp32: output queue overflow (credit bug)")
  outQ.io.deq <> ext_data_o

  val deq = outQ.io.deq.fire
  when(fire =/= deq) { credit := Mux(fire, credit - 1.U, credit + 1.U) }

  ext_busy_o := credit =/= Qdepth.U
}
