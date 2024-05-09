#include "dfcir/DFCIRDialect.h"
#include "dfcir/DFCIRTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_TYPEDEF_CLASSES

#include "dfcir/DFCIRTypes.cpp.inc"

void mlir::dfcir::DFCIRDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST

#include "dfcir/DFCIRTypes.cpp.inc"

  >();
}
