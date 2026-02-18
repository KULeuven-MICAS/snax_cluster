package snax.utils

import chisel3._
import chisel3.util._

class CriticalLoopFinder(temporalDimension: Int, addressWidth: Int, fixedCacheDepth: Int) extends Module {
    val io = IO(new Bundle {
        val temporalBounds    = Input(Vec(temporalDimension, UInt(addressWidth.W)))
        val temporalStrides   = Input(Vec(temporalDimension, UInt(addressWidth.W)))
        val criticalLoop = Output(UInt(log2Ceil(temporalDimension).W))
        val fixedCachePeriod = Output(UInt(log2Ceil(fixedCacheDepth).W))
        val anyLoopFound = Output(Bool())
        val totalBounds = Output(Vec(temporalDimension, UInt(addressWidth.W)))
    })
    
    // Compute totalBounds (this part is fine)
    for (i <- 0 until temporalDimension) {
        if (i == 0)
        io.totalBounds(i) := io.temporalBounds(i)
        else
            when(io.temporalStrides(i) === 0.U) {
                io.totalBounds(i) := io.totalBounds(i-1)
            } .otherwise {
                io.totalBounds(i) := io.totalBounds(i-1) * io.temporalBounds(i)
            }
    }
    dontTouch(io.totalBounds)
    dontTouch(io.temporalStrides)
    dontTouch(io.temporalBounds)

    // Boolean condition per index (reversed)
    val cond = (0 until temporalDimension).reverse.map { i =>
        if (i == 0) {
            false.B
        } else {
            (io.totalBounds(i-1) < fixedCacheDepth.U) && (io.temporalStrides(i) === 0.U)
        }
    }

    io.anyLoopFound := cond.reduce(_ || _)

    // Find first index where cond is true
    val reversedIdx = PriorityEncoder(cond)

    // Convert back to forward index
    io.criticalLoop := (temporalDimension.U - 1.U) - reversedIdx

    io.fixedCachePeriod := io.totalBounds(io.criticalLoop - 1.U)
}
