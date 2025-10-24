// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Xiaoling Yi <xiaoling.yi@kuleuven.be>
// Yunhao Deng <yunhao.deng@kuleuven.be>

package snax_acc.utils

import chisel3._

class BasicCounter(width: Int, hasCeil: Boolean = true, nameTag: String = "Default")
    extends Module
    with RequireAsyncReset {

  val io = IO(new Bundle {
    val tick    = Input(Bool())
    val reset   = Input(Bool())
    // Only include `ceil` if hasCeil is true
    val ceilOpt = if (hasCeil) Some(Input(UInt(width.W))) else None

    val value   = Output(UInt(width.W))
    val lastVal = Output(Bool())
  })

  override def desiredName: String = s"BasicCounter_${nameTag}_${width}_${hasCeil}"

  val nextValue = Wire(UInt(width.W))
  val value     = RegNext(nextValue, 0.U)

  if (hasCeil) {
    val ceil = io.ceilOpt.get
    nextValue  := Mux(
      io.reset,
      0.U,
      Mux(io.tick, Mux(value < ceil - 1.U, value + 1.U, 0.U), value)
    )
    io.value   := value
    io.lastVal := (value === ceil - 1.U) && io.tick
  } else {
    nextValue  := Mux(io.reset, 0.U, Mux(io.tick, value + 1.U, value))
    io.value   := value
    io.lastVal := value.andR && io.tick
  }
}

class NestCounter(width: Int, loopNum: Int, hasCeil: Boolean = true) extends Module with RequireAsyncReset {

  val io = IO(new Bundle {
    val tick    = Input(Bool())
    val reset   = Input(Bool())
    // Only include ceil vector if hasCeil = true
    val ceilOpt = if (hasCeil) Some(Input(Vec(loopNum, UInt(width.W)))) else None

    val value   = Output(Vec(loopNum, UInt(width.W)))
    val lastVal = Output(Vec(loopNum, Bool()))
  })

  val counter = Seq.fill(loopNum)(Module(new BasicCounter(width, hasCeil)))

  // Reset connections
  counter.foreach(_.io.reset := io.reset)

  // Connect ceil signals only if enabled
  if (hasCeil) {
    val ceilVec = io.ceilOpt.get
    counter.zipWithIndex.foreach { case (c, i) =>
      c.io.ceilOpt.get := ceilVec(i)
    }
  }

  // Tick chaining
  counter(0).io.tick := io.tick
  counter.zip(counter.tail).foreach { case (c1, c2) =>
    c2.io.tick := c1.io.lastVal && io.tick
  }

  io.value   := VecInit(counter.map(_.io.value))
  io.lastVal := VecInit(counter.map(_.io.lastVal))
}

object NestCounterEmitter extends App {
  val width   = 4
  val loopNum = 3
  emitVerilog(
    new NestCounter(width, loopNum),
    Array("--target-dir", "generated/SpatialArray")
  )
}
