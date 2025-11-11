package snax.utils

import chisel3._
import chisel3.util._

class CriticalLoopFinder(temporalDimension: Int, addressWidth: Int, fixedCacheDepth: Int) extends Module {
    val io = IO(new Bundle {
        val temporalBounds    = Input(Vec(temporalDimension, UInt(addressWidth.W)))
        val temporalStrides   = Input(Vec(temporalDimension, UInt(addressWidth.W)))
        val criticalLoop = Output(UInt(log2Ceil(temporalDimension).W))
        val fixedCachePeriod = Output(UInt(log2Ceil(fixedCacheDepth).W))
    })
    
    // Compute totalBounds (this part is fine)
    val totalBounds = Wire(Vec(temporalDimension, UInt(addressWidth.W)))
    for (i <- 0 until temporalDimension) {
        if (i == 0)
        totalBounds(i) := io.temporalBounds(i)
        else
        totalBounds(i) := totalBounds(i-1) * io.temporalBounds(i)
    }

    // Boolean condition per index (reversed)
    val cond = (totalBounds.reverse.zip(io.temporalStrides.reverse)).map { case (b, s) =>
        (b < fixedCacheDepth.U) && (s === 0.U)
    }

    // Find first index where cond is true
    val reversedIdx = PriorityEncoder(cond)

    // Convert back to forward index
    io.criticalLoop := (temporalDimension.U - 1.U) - reversedIdx

    io.fixedCachePeriod := totalBounds(io.criticalLoop - 1.U)
}
