// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

package snax.DataPathExtension

import chisel3._
import chisel3.util._

import snax.utils.RegQueue

/** CrestDecompressor: decoder of CREST (Centre-Relative Encoding with Split Tiers), the lossless multi-format weight
  * format; one output beat per cycle.
  *
  * The format is the one `hw/chisel/doc/crest_decompressor/crest_codec.py` encodes (spec: crest_decompressor.md). The output is
  * the weight tensor as stored, in beats of L lanes:
  *
  *   - mode 0, nib: 128 lanes of 4 bits (INT4 / MXFP4 / NVFP4 nibbles)
  *   - mode 1, byte: 64 lanes of 8 bits (INT8 / FP8)
  *   - mode 2, bf16: 32 lanes of 16 bits; the exponent byte is coded, sign and mantissa travel raw
  *
  * Word 0 of the stream configures the transfer: [1:0] mode, [2] keymap, [10:3] centre, [13:11] p, [16:14] e1. A lane's
  * p-bit plane code is a KEY (keys < K0 = 2^p - 2 are direct), K0 = a tier-1 escape (an e1-bit entry, key = K0 + entry)
  * or K0 + 1 = a tier-2 escape (the raw symbol). The key orders symbols outward from the centre: keymap 0 = zigzag of
  * the two's-complement distance, keymap 1 = sign-magnitude, 2 * zigzag(|m| - centre) + sign. No table: the inverse is
  * an adder per lane.
  *
  * Groups of `groupBeats` beats: coded = E escape words (32-bit header: [14:0] E, [15] 0, [31:16] n1; then n1 tier-1
  * entries, then the tier-2 entries, lane order) + ceil(gb * L * lb / 512) plane words, lb = p (+ 8 raw bits in bf16);
  * raw = a header word with bit 15 set + the gb beats. CSR 0 = N output beats.
  *
  * DATAPATH:
  *   - Router: stream header, then HDR -> ESC -> PLANE (or RAW) per group, two escape buffers so groups overlap.
  *   - Plane window A, B and a plane bit pointer. Every lane field width times L is a multiple of 32 bits, so a beat's
  *     slice is a 16:1 funnel over the 2-word window at 32-bit steps.
  *   - Stage 1: each lane's field for the configured (mode, p), its class, two prefix popcounts (tier-1 and tier-2
  *     entry numbers), and two escape windows: the escape buffer shifted by the tier-1 and tier-2 pointers. Both
  *     pointer updates depend only on plane-derived counts.
  *   - Stage 2: escaped lanes take their entries (K1 / K2 per pass), every key is mapped back to its symbol, and the
  *     lanes are assembled in the mode's width.
  *
  * A beat with more than `k1Lanes` tier-1 or `k2Lanes` tier-2 escapes takes extra passes. On real checkpoints (GPTQ
  * INT4, FP8, INT8, BF16) at 32 / 16 that is 0-1.4% of the beats.
  */
class CrestDecompressor(groupBeats: Int, escWords: Int, k1Lanes: Int, k2Lanes: Int, planeDepth: Int)(implicit
  extensionParam: DataPathExtensionParam
) extends DataPathExtension {

  val W   = extensionParam.dataWidth
  val S   = W / 4 // lane slots: lanes of the nibble mode
  val HB  = 32    // group header bits
  val G   = groupBeats
  val K1  = k1Lanes
  val K2  = k2Lanes
  val EBW = escWords * W

  require(W == 512, "CrestDecompressor: built for 512-bit beats")
  require(G          >= 2  && G <= 255, s"CrestDecompressor: groupBeats ($G) must be in [2, 255]")
  require(escWords   >= 1, "CrestDecompressor: escWords must be at least 1")
  require(isPow2(K1) && K1 >= 2 && K1 <= S && isPow2(K2) && K2 >= 2 && K2 <= S, "CrestDecompressor: k1/k2Lanes")
  require(planeDepth >= 2, "CrestDecompressor: planeDepth must be at least 2")

  val lgK1 = log2Ceil(K1)
  val lgK2 = log2Ceil(K2)
  val gbW  = log2Ceil(G + 1)
  val cntW = log2Ceil((G max escWords) + 1)
  val nplW = log2Ceil(G + 1)        // a group's plane words never exceed its beats
  val jW   = log2Ceil(G)
  val nW   = log2Ceil(S + 1)
  val subW = log2Ceil(S / K2) max 1 // the longest beat: S tier-2 escapes, K2 a pass
  val qW   = log2Ceil(EBW + 1) + 4  // escape pointers, with headroom for a bad header

  // The supported (mode, p): nib p 2..3, byte p 2..7, bf16 p 2..7.
  case class Cfg(mode: Int, p: Int) {
    val lanes = Seq(128, 64, 32)(mode)
    val lb    = p + (if (mode == 2) 8 else 0)
    val bpb   = lanes * lb // plane bits per beat, a multiple of 32
  }
  val cfgs = (2 to 3).map(Cfg(0, _)) ++ (2 to 7).map(Cfg(1, _)) ++ (2 to 7).map(Cfg(2, _))

  def minG(x: UInt): UInt = Mux(x >= G.U, G.U(gbW.W), x(gbW - 1, 0))

  // ---------------------------------------------------------------------------------------------------------------
  // Transfer configuration (stream header)
  // ---------------------------------------------------------------------------------------------------------------
  val cMode   = RegInit(0.U(2.W))
  val cKeymap = RegInit(0.U(1.W))
  val cCentre = RegInit(0.U(8.W))
  val cP      = RegInit(2.U(3.W))
  val cE1     = RegInit(1.U(3.W))
  val cfgSel  = VecInit(cfgs.map(c => cMode === c.mode.U && cP === c.p.U))
  val bpb     = Mux1H(cfgSel, cfgs.map(c => c.bpb.U(10.W))) // plane bits per beat (<= 480)
  val sbBits  = Mux(cMode === 0.U, 4.U(4.W), 8.U(4.W))      // tier-2 entry width = symbol width
  val laneCnt = Mux1H(Seq(cMode === 0.U, cMode === 1.U, cMode === 2.U), Seq(128.U(8.W), 64.U(8.W), 32.U(8.W)))
  val k0      = ((1.U(8.W) << cP) - 2.U)(6, 0)              // 2^p - 2: tier-1 code; K0 + 1: tier-2
  def nplOf(gb: UInt): UInt = ((gb * bpb) + 511.U) >> 9

  // ---------------------------------------------------------------------------------------------------------------
  // Router
  // ---------------------------------------------------------------------------------------------------------------
  object RState extends ChiselEnum { val sStream, sHdr, sEsc, sPlane, sRaw = Value }

  class GroupInfo extends Bundle {
    val raw = Bool()
    val buf = UInt(1.W)
    val n1  = UInt(16.W)
  }

  val rState   = RegInit(RState.sStream)
  val inRemain = RegInit(0.U(32.W))
  val rGb      = RegInit(0.U(gbW.W))
  val rCnt     = RegInit(0.U(cntW.W))
  val rEscIdx  = RegInit(0.U(log2Ceil(escWords max 2).W))
  val rN1      = RegInit(0.U(16.W))
  val fillBuf  = RegInit(0.U(1.W))
  val escBuf   = Reg(Vec(2, Vec(escWords, UInt(W.W))))
  val escValid = RegInit(VecInit(Seq.fill(2)(false.B)))

  val grpQ   = Module(new RegQueue(new GroupInfo, 2, hasFlush = true))
  val planeQ = Module(new RegQueue(UInt(W.W), planeDepth, hasFlush = true))

  val word   = ext_data_i.bits
  val hdrRaw = word(15)
  val hdrE   = word(14, 0)
  val hdrN1  = word(31, 16)
  val gbIn   = minG(inRemain)

  ext_data_i.ready     := false.B
  grpQ.io.enq.valid    := false.B
  grpQ.io.enq.bits.raw := false.B
  grpQ.io.enq.bits.buf := fillBuf
  grpQ.io.enq.bits.n1  := rN1
  planeQ.io.enq.valid  := false.B
  planeQ.io.enq.bits   := word

  def finishEscape(gb: UInt, n1: UInt): Unit = {
    grpQ.io.enq.valid   := true.B
    grpQ.io.enq.bits.n1 := n1
    escValid(fillBuf)   := true.B
    fillBuf             := ~fillBuf
    rState              := RState.sPlane
    rCnt                := nplOf(gb)
  }

  switch(rState) {
    is(RState.sStream) {
      ext_data_i.ready := inRemain =/= 0.U
      when(ext_data_i.fire) {
        val m = word(1, 0)
        val p = word(13, 11)
        assert(
          (m === 0.U && p >= 2.U && p <= 3.U && word(16, 14) >= 1.U && word(16, 14) <= 3.U) ||
            ((m === 1.U || m === 2.U) && p >= 2.U && word(16, 14) >= 1.U),
          "CrestDecompressor: unsupported stream header (mode, p, e1)"
        )
        cMode                         := m
        cKeymap                       := word(2)
        cCentre                       := word(10, 3)
        cP                            := p
        cE1                           := word(16, 14)
        rState                        := RState.sHdr
      }
    }
    is(RState.sHdr) {
      ext_data_i.ready := inRemain =/= 0.U && grpQ.io.enq.ready && !escValid(fillBuf)
      when(ext_data_i.fire) {
        inRemain := inRemain - gbIn
        rGb      := gbIn
        when(hdrRaw) {
          assert((word & ~(1.U(W.W) << 15)) === 0.U, "CrestDecompressor: a raw header word must have only bit 15 set")
          grpQ.io.enq.valid    := true.B
          grpQ.io.enq.bits.raw := true.B
          rState               := RState.sRaw
          rCnt                 := gbIn
        }.otherwise {
          assert(hdrE =/= 0.U && hdrE <= escWords.U, "CrestDecompressor: coded header with E = 0 or E > escWords")
          escBuf(fillBuf)(0)  := word
          rN1                 := hdrN1
          when(hdrE <= 1.U) {
            finishEscape(gbIn, hdrN1)
          }.otherwise {
            rState  := RState.sEsc
            rCnt    := Mux(hdrE > escWords.U, (escWords - 1).U, hdrE - 1.U)
            rEscIdx := 1.U
          }
        }
      }
    }
    is(RState.sEsc) {
      val lastEsc = rCnt === 1.U
      ext_data_i.ready := !lastEsc || grpQ.io.enq.ready
      when(ext_data_i.fire) {
        escBuf(fillBuf)(rEscIdx) := word
        rEscIdx                  := rEscIdx + 1.U
        rCnt                     := rCnt - 1.U
        when(lastEsc)(finishEscape(rGb, rN1))
      }
    }
    is(RState.sPlane, RState.sRaw) {
      ext_data_i.ready    := planeQ.io.enq.ready
      planeQ.io.enq.valid := ext_data_i.valid
      when(ext_data_i.fire) {
        rCnt                      := rCnt - 1.U
        when(rCnt === 1.U)(rState := RState.sHdr)
      }
    }
  }

  assert(
    !(ext_data_i.valid && rState === RState.sHdr && inRemain === 0.U && !ext_start_i),
    "CrestDecompressor: input word after the last group of the transfer"
  )

  // ---------------------------------------------------------------------------------------------------------------
  // Decoder
  // ---------------------------------------------------------------------------------------------------------------
  val curValid  = RegInit(false.B)
  val curRaw    = RegInit(false.B)
  val curBuf    = RegInit(0.U(1.W))
  val gbDec     = RegInit(0.U(gbW.W))
  val nplDec    = RegInit(0.U(nplW.W))
  val j         = RegInit(0.U(jW.W))
  val sub       = RegInit(0.U(subW.W))
  val loadCnt   = RegInit(0.U(nplW.W))
  val outRemain = RegInit(0.U(32.W))
  val pp        = RegInit(0.U(9.W))   // plane bit pointer into word A, a multiple of 32
  val q1        = RegInit(HB.U(qW.W)) // tier-1 entry pointer (bits into the escape buffer)
  val q2        = RegInit(HB.U(qW.W)) // tier-2 entry pointer
  val winA      = RegInit(0.U(W.W))
  val winB      = RegInit(0.U(W.W))
  val aV        = RegInit(false.B)
  val bV        = RegInit(false.B)

  val outQ = Module(new RegQueue(UInt(W.W), 2, hasFlush = true))

  // ---- stage 1: the beat's plane slice and lane fields ----
  val needB  = (pp +& bpb) > W.U
  val window = Cat(winB, winA)
  val slice  = VecInit((0 until 16).map(k => window(32 * k + 479, 32 * k)))(pp(8, 5))

  def laneField(i: Int, width: Int): Seq[(Cfg, UInt)] =
    cfgs.filter(c => i < c.lanes).map { c =>
      val lo = i * c.lb
      (c, slice(lo + c.lb - 1, lo).pad(width))
    }

  val code    = VecInit((0 until S).map { i =>
    val opts = laneField(i, 15)
    Mux1H(opts.map(o => cfgSel(cfgs.indexOf(o._1))), opts.map(o => o._2(o._1.p - 1, 0).pad(7)))
  })
  val rawByte = VecInit((0 until 32).map { i =>
    val opts = laneField(i, 15).filter(_._1.mode == 2)
    Mux1H(opts.map(o => cfgSel(cfgs.indexOf(o._1))), opts.map(o => (o._2 >> o._1.p)(7, 0)))
  })
  val active  = VecInit((0 until S).map(i => i.U < laneCnt))
  val f1      = VecInit((0 until S).map(i => active(i) && code(i) === k0))
  val f2      = VecInit((0 until S).map(i => active(i) && code(i) === k0 + 1.U))

  def prefix(f: Seq[Bool]): Seq[UInt] = {
    var v: Vector[UInt] = f.map(e => Mux(e, 1.U(nW.W), 0.U(nW.W))).toVector
    var span = 1
    while (span < S) {
      val s    = span
      val prev = v
      v = prev.zipWithIndex.map { case (x, i) => if ((i / s) % 2 == 1) (x + prev((i / s) * s - 1))(nW - 1, 0) else x }
      span *= 2
    }
    v
  }
  val incl1 = prefix(f1)
  val incl2 = prefix(f2)
  val idx1  = (0 until S).map(i => if (i == 0) 0.U(nW.W) else incl1(i - 1))
  val idx2  = (0 until S).map(i => if (i == 0) 0.U(nW.W) else incl2(i - 1))
  val n1b   = incl1(S - 1)
  val n2b   = incl2(S - 1)

  val base1    = sub << lgK1
  val base2    = sub << lgK2
  val rem1     = Mux(n1b > base1, n1b - base1, 0.U)
  val rem2     = Mux(n2b > base2, n2b - base2, 0.U)
  val lastPass = rem1 <= K1.U && rem2 <= K2.U
  val take1    = Mux(rem1 > K1.U, K1.U, rem1)
  val take2    = Mux(rem2 > K2.U, K2.U, rem2)
  val in1      = VecInit((0 until S).map(i => f1(i) && (idx1(i) >> lgK1) === sub))
  val in2      = VecInit((0 until S).map(i => f2(i) && (idx2(i) >> lgK2) === sub))

  // ---- stage 1: the escape windows ----
  val escCat = Mux(curBuf.asBool, Cat(escBuf(1).reverse), Cat(escBuf(0).reverse))
  val win1   = (escCat >> q1)(7 * K1 - 1, 0)
  val win2   = (escCat >> q2)(8 * K2 - 1, 0)
  val ent1   = VecInit((0 until K1).map { k =>
    MuxLookup(cE1, 0.U(7.W))((1 to 7).map(e => e.U -> win1(k * e + e - 1, k * e).pad(7)))
  })
  val ent2   = VecInit((0 until K2).map(k => Mux(cMode === 0.U, win2(4 * k + 3, 4 * k).pad(8), win2(8 * k + 7, 8 * k))))

  // ---- stage-2 registers ----
  val s2Valid = RegInit(false.B)
  val s2Last  = RegInit(false.B)
  val s2Code  = RegInit(VecInit(Seq.fill(S)(0.U(7.W))))
  val s2F1    = RegInit(0.U(S.W))
  val s2F2    = RegInit(0.U(S.W))
  val s2In1   = RegInit(0.U(S.W))
  val s2In2   = RegInit(0.U(S.W))
  val s2Loc1  = RegInit(VecInit(Seq.fill(S)(0.U(lgK1.W))))
  val s2Loc2  = RegInit(VecInit(Seq.fill(S)(0.U(lgK2.W))))
  val s2Raw   = RegInit(VecInit(Seq.fill(32)(0.U(8.W))))
  val s2Ent1  = RegInit(VecInit(Seq.fill(K1)(0.U(7.W))))
  val s2Ent2  = RegInit(VecInit(Seq.fill(K2)(0.U(8.W))))
  val acc     = RegInit(VecInit(Seq.fill(S)(0.U(8.W)))) // lanes placed by earlier passes

  val s2Consume = s2Valid && (!s2Last || outQ.io.enq.ready)
  val s2Free    = !s2Valid || s2Consume

  val codedActive = curValid    && !curRaw
  val issue       = codedActive && aV     && (!needB || bV)      && s2Free
  val issueLast   = issue       && lastPass
  val beatIsLast  = j === gbDec - 1.U
  val rawFire     = curValid    && curRaw && planeQ.io.deq.valid && !s2Valid && outQ.io.enq.ready

  val groupEndCoded = issueLast                 && beatIsLast
  val groupEnd      = groupEndCoded || (rawFire && beatIsLast)

  // ---- stage 2: keys back to symbols, lanes assembled ----
  def unzig(k: UInt, w: Int):      SInt = { // 0, 1, 2, 3, ... -> 0, -1, +1, -2, ... as a (w+1)-bit SInt
    val kk  = k.pad(w)(w - 1, 0)
    val pos = (kk >> 1).pad(w + 1)
    val neg = (0.U((w + 1).W) - ((kk +& 1.U) >> 1).pad(w + 1))(w, 0)
    Mux(kk(0), neg, pos).asSInt
  }
  def symbolOf(key: UInt, w: Int): UInt = {
    val iv  = (cCentre.zext + unzig(key, w)).asUInt(w - 1, 0)
    val mag = (cCentre.zext + unzig(key >> 1, w)).asUInt(w - 2, 0)
    val smv = Cat(key(0), mag)
    Mux(cKeymap === 0.U, iv, smv)(w - 1, 0)
  }

  val laneSym  = VecInit((0 until S).map { i =>
    val e1v  = s2Ent1(s2Loc1(i))
    val key  = Mux(s2F1(i), (k0 +& e1v)(7, 0), s2Code(i).pad(8))
    val sym8 = symbolOf(key, 8)
    val sym4 = symbolOf(key(3, 0), 4).pad(8)
    val sym  = if (i < 64) Mux(cMode === 0.U, sym4, sym8) else sym4
    val t2   = s2Ent2(s2Loc2(i))
    val v    = Mux(s2F2(i), t2, sym)
    Mux(s2In1(i) || s2In2(i), v, Mux(s2F1(i) || s2F2(i), acc(i), v))
  })
  val beatNib  = Cat((0 until 128).reverse.map(i => laneSym(i)(3, 0)))
  val beatByte = Cat((0 until 64).reverse.map(i => laneSym(i)))
  val beatBf16 = Cat((0 until 32).reverse.map(i => Cat(s2Raw(i)(7), laneSym(i), s2Raw(i)(6, 0))))
  val beatOut  = MuxLookup(cMode, beatNib)(Seq(1.U -> beatByte, 2.U -> beatBf16))

  outQ.io.enq.valid            := (s2Valid && s2Last) || rawFire
  outQ.io.enq.bits             := Mux(rawFire, planeQ.io.deq.bits, beatOut)
  ext_data_o <> outQ.io.deq
  when(s2Valid && !s2Last)(acc := laneSym)

  when(issue) {
    s2Valid := true.B
    s2Last  := lastPass
    s2Code  := code
    s2F1    := f1.asUInt
    s2F2    := f2.asUInt
    s2In1   := in1.asUInt
    s2In2   := in2.asUInt
    s2Loc1  := VecInit(idx1.map(_(lgK1 - 1, 0)))
    s2Loc2  := VecInit(idx2.map(_(lgK2 - 1, 0)))
    s2Raw   := rawByte
    s2Ent1  := ent1
    s2Ent2  := ent2
  }.elsewhen(s2Consume) {
    s2Valid := false.B
  }

  when(issue) {
    q1 := q1 + take1 * cE1
    q2 := q2 + take2 * sbBits
    when(lastPass) {
      sub := 0.U
      j   := j + 1.U
    }.otherwise {
      sub := sub + 1.U
    }
  }
  when(rawFire)(j                      := j + 1.U)
  when(groupEndCoded)(escValid(curBuf) := false.B)

  // ---- group advance ----
  val nextInfo = grpQ.io.deq.bits
  val startGrp = (!curValid || groupEnd) && grpQ.io.deq.valid
  val gbNew    = minG(outRemain)
  val nplNew   = nplOf(gbNew)
  grpQ.io.deq.ready := startGrp

  // ---- plane window: the bit pointer advances by bpb a beat; crossing 512 retires word A ----
  val ppNext = pp +& bpb
  val retire = issueLast && !beatIsLast && ppNext >= W.U
  when(issueLast && !beatIsLast)(pp := Mux(ppNext >= W.U, ppNext - W.U, ppNext))

  when(startGrp) {
    curValid  := true.B
    curRaw    := nextInfo.raw
    curBuf    := nextInfo.buf
    gbDec     := gbNew
    nplDec    := nplNew
    outRemain := outRemain - gbNew
    j         := 0.U
    sub       := 0.U
    q1        := HB.U
    q2        := HB.U +& nextInfo.n1 * cE1
    pp        := 0.U
  }.elsewhen(groupEnd) {
    curValid := false.B
  }

  val aV1 = Mux(groupEnd, false.B, Mux(retire, bV, aV))
  val bV1 = Mux(groupEnd || retire, false.B, bV)
  val a1  = Mux(retire, winB, winA)

  val loadCoded  = Mux(startGrp, !nextInfo.raw, codedActive && !groupEnd)
  val loadCntEff = Mux(startGrp, 0.U, loadCnt)
  val nplEff     = Mux(startGrp, nplNew, nplDec)
  val load       = !rawFire && loadCoded && loadCntEff < nplEff && (!aV1 || !bV1) && planeQ.io.deq.valid
  planeQ.io.deq.ready := load || rawFire

  winA    := Mux(load && !aV1, planeQ.io.deq.bits, a1)
  winB    := Mux(load && aV1, planeQ.io.deq.bits, winB)
  aV      := aV1 || (load && !aV1)
  bV      := bV1 || (load && aV1)
  loadCnt := loadCntEff + load.asUInt

  // ---------------------------------------------------------------------------------------------------------------
  // Start
  // ---------------------------------------------------------------------------------------------------------------
  grpQ.io.flush.get   := ext_start_i
  planeQ.io.flush.get := ext_start_i
  outQ.io.flush.get   := ext_start_i

  when(ext_start_i) {
    rState    := RState.sStream
    inRemain  := ext_csr_i(0)
    outRemain := ext_csr_i(0)
    fillBuf   := 0.U
    escValid  := VecInit(Seq.fill(2)(false.B))
    curValid  := false.B
    aV        := false.B
    bV        := false.B
    s2Valid   := false.B
    j         := 0.U
    sub       := 0.U
    loadCnt   := 0.U
    pp        := 0.U
  }

  ext_busy_o := inRemain =/= 0.U || rState =/= RState.sHdr || outRemain =/= 0.U || curValid || s2Valid ||
    outQ.io.deq.valid
}

/** `groupBeats` and `escWords` are the format the encoder used (crest_codec.py: G = 64, EMAX = 12), so a cfg has to state
  * them. `k1Lanes` / `k2Lanes` (escape entries placed per cycle) and `planeDepth` only trade area against cycles.
  * `planeDepth` = 8 lets the router read a group's (up to 12) escape words while the previous group's last beats still
  * decode; at 4 the FP8 and INT8 checkpoints lose ~4 cycles per group boundary.
  */
class HasCrestDecompressor(
  groupBeats: Int,
  escWords:   Int,
  k1Lanes:    Int = 32,
  k2Lanes:    Int = 16,
  planeDepth: Int = 8,
  dataWidth:  Int = 512
) extends HasDataPathExtension {
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName = "CrestDecompressor",
      userCsrNum = 1,
      dataWidth  = dataWidth
    )

  def instantiate(clusterName: String): CrestDecompressor =
    Module(new CrestDecompressor(groupBeats, escWords, k1Lanes, k2Lanes, planeDepth) {
      override def desiredName = clusterName + namePostfix
    })
}
