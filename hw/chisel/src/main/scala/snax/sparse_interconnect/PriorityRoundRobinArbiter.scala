package snax.sparse_interconnect

import chisel3._
import chisel3.util._

class PriorityRequest(priorityWidth: Int) extends Bundle {
  val valid    = Bool()
  val priority = UInt(priorityWidth.W) // Example width for priority
}

class PriorityRoundRobinArbiter(NumInp: Int, priorityWidth: Int) extends Module {

  val io = IO(new Bundle {
    val requests  = Input(Vec(NumInp, new PriorityRequest(priorityWidth)))
    val selection = Decoupled(UInt(log2Ceil(NumInp).W))
  })

  // Instantiate normal round-robin arbiter
  val arbiter = Module(new RoundRobinArbiter(NumInp))

  // Determine the highest valid priority among the requests
  val effectivePriorities = io.requests.map(req => Mux(req.valid, req.priority, 0.U))
  val maxPriority         = effectivePriorities.reduce((a, b) => Mux(a > b, a, b))

  // Mask requests that do not have the highest priority
  for (i <- 0 until NumInp) {
    arbiter.io.requests(i) := io.requests(i).valid && (io.requests(i).priority === maxPriority)
  }

  // Connect arbiter output to module output
  io.selection <> arbiter.io.selection

}
