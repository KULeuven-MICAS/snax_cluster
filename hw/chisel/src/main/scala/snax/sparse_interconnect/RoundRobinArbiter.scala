package snax.sparse_interconnect

import chisel3._
import chisel3.util._

class RoundRobinArbiter(NumInp: Int) extends Module {

  val selWidth: Int = log2Ceil(NumInp)

  val io = IO(new Bundle {
    val requests  = Input(Vec(NumInp, Bool()))
    val selection = Decoupled(UInt(selWidth.W))
  })

  // keep track of previous selection for round-robin arbitration
  // initialize to the last input to start with the first input
  val previous = RegNext(io.selection.bits, (NumInp - 1).U)

  // lock on previous request until success
  val lock = RegInit(false.B)

  // Arbitration and request routing
  // Collect all valid requests to this bank:
  val validRequests = io.requests
  val anyValid: Bool = validRequests.reduce(_ || _)

  // Collect all next requests (requests next in line for round-robin fashion)
  val nextRequests = validRequests.zipWithIndex.map { case (valid, idx) =>
    Mux(idx.U > previous, valid, false.B)
  }
  val nextValid: Bool = nextRequests.reduce(_ || _)

  // On lock, keep sending the previous request if it is still valid.
  // If not, take the next valid request if available in round-robin fashion.
  // Otherwise: wrap around and take the first valid request.
  val selectedRequest = Wire(UInt(selWidth.W))
  when(lock && validRequests(previous)) {
    selectedRequest := previous
  }.elsewhen(nextValid) {
    selectedRequest := PriorityEncoder(nextRequests)
  }.otherwise {
    selectedRequest := PriorityEncoder(validRequests)
  }

  // Lock on unsuccessfull request
  lock := io.selection.valid && !io.selection.ready

  // Set outputs
  io.selection.bits  := selectedRequest
  io.selection.valid := anyValid

}
