#include "dfcir/DFCIRDialect.h"
#include "dfcir/DFCIRTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "dfcir/DFCIRTypeInterfaces.cpp.inc" // Cannot enforce header sorting.

mlir::Type mlir::dfcir::DFCIRStreamType::getDFType() {
  return getStreamType();
}

mlir::Type mlir::dfcir::DFCIRScalarType::getDFType() {
  return getScalarType();
}

mlir::Type mlir::dfcir::DFCIRConstantType::getDFType() {
  return getConstType();
}

#define GET_TYPEDEF_CLASSES

#include "dfcir/DFCIRTypes.cpp.inc"

void mlir::dfcir::DFCIRDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST

#include "dfcir/DFCIRTypes.cpp.inc"

  >();
}
