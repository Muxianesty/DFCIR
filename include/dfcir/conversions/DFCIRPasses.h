#ifndef DFCIR_PASSES_H
#define DFCIR_PASSES_H

#include "dfcir/DFCIROperations.h"
#include "mlir/Pass/Pass.h"

#include "memory"

namespace mlir::dfcir {

enum Ops {
  UNDEFINED,
  ADD_INT,
  ADD_FLOAT,
  MUL_INT,
  MUL_FLOAT
};

} // namespace mlir::dfcir

typedef std::unordered_map<mlir::dfcir::Ops, unsigned> LatencyConfig;


namespace mlir::dfcir {

using std::unique_ptr;
using mlir::Pass;

unique_ptr<Pass> createDFCIRToFIRRTLPass(LatencyConfig *config = nullptr);

unique_ptr<Pass> createDFCIRDijkstraSchedulerPass();

unique_ptr<Pass> createDFCIRLinearSchedulerPass();

} // namespace mlir::dfcir

#define GEN_PASS_REGISTRATION

#include "dfcir/conversions/DFCIRPasses.h.inc"

#endif // DFCIR_PASSES_H
