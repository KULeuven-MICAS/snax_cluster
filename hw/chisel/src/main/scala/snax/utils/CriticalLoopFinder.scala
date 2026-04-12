package snax.utils

import chisel3._
import chisel3.util._

class CriticalLoopFinder(temporalDimension: Int, addressWidth: Int, fixedCacheDepth: Int) extends Module {
    val tbWidth = log2Ceil(fixedCacheDepth) + 1

    val io = IO(new Bundle {
        val temporalBounds    = Input(Vec(temporalDimension, UInt(addressWidth.W)))
        val temporalStrides   = Input(Vec(temporalDimension, UInt(addressWidth.W)))
        val criticalLoop = Output(UInt(log2Ceil(temporalDimension).W))
        val fixedCachePeriod = Output(UInt(tbWidth.W))
        val anyLoopFound = Output(Bool())
        val totalBounds = Output(Vec(temporalDimension, UInt(tbWidth.W)))
    })

    // Overflow flags: once set, all outer dimensions propagate it
    val overflow = Wire(Vec(temporalDimension, Bool()))

    // Compute totalBounds with reduced width, tracking overflow.
    // To avoid a full tbWidth × addressWidth multiplier, we split temporalBounds
    // into low (tbWidth bits) and high parts. The high part only contributes to
    // bits >= tbWidth, so:
    //   - totalBounds(i) = (totalBounds(i-1) * bLow)(tbWidth-1, 0)  (identical to full product's low bits)
    //   - overflow if bHigh != 0 && totalBounds(i-1) != 0, or if the small product overflows tbWidth bits
    for (i <- 0 until temporalDimension) {
        if (i == 0) {
            io.totalBounds(i) := io.temporalBounds(i)(tbWidth - 1, 0)
            overflow(i) := io.temporalBounds(i)(addressWidth - 1, tbWidth).orR
        } else {
            when(io.temporalStrides(i) === 0.U) {
                io.totalBounds(i) := io.totalBounds(i-1)
                overflow(i) := overflow(i-1)
            } .otherwise {
                val a = io.totalBounds(i-1)                         // tbWidth bits
                val bLow  = io.temporalBounds(i)(tbWidth - 1, 0)   // tbWidth bits
                val bHigh = io.temporalBounds(i)(addressWidth - 1, tbWidth)

                // Small tbWidth × tbWidth multiplier → 2*tbWidth bits
                val smallProduct = a * bLow
                io.totalBounds(i) := smallProduct(tbWidth - 1, 0)

                // Overflow: propagated, or high part guarantees it, or small product exceeds tbWidth bits
                val smallProductOverflow = smallProduct(2 * tbWidth - 1, tbWidth).orR
                overflow(i) := overflow(i-1) || (bHigh.orR && a.orR) || smallProductOverflow
            }
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
            !overflow(i-1) && (io.totalBounds(i-1) <= fixedCacheDepth.U) && (io.temporalStrides(i) === 0.U) && (io.temporalBounds(i) > 1.U)
        }
    }

    io.anyLoopFound := cond.reduce(_ || _)

    // Find first index where cond is true
    val reversedIdx = PriorityEncoder(cond)

    // Convert back to forward index
    io.criticalLoop := (temporalDimension.U - 1.U) - reversedIdx

    io.fixedCachePeriod := io.totalBounds(io.criticalLoop - 1.U)
}
