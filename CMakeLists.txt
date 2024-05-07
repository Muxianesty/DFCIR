cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(DFCIR LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(DEFINED CIRCT_BIN_HOME)
  message(STATUS "CIRCT_BIN_HOME: ${CIRCT_BIN_HOME}")
  list(APPEND CMAKE_PREFIX_PATH "${CIRCT_BIN_HOME}")
else()
  message(FATAL_ERROR "CIRCT_BIN_HOME var is not set!")
endif()

find_package(LpSolve REQUIRED)

unset(LLVM_DIR)
unset(MLIR_DIR)
unset(CIRCT_DIR)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(CIRCT REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_CMAKE_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_CMAKE_DIR}")
message(STATUS "Using CIRCTConfig.cmake in: ${CIRCT_CMAKE_DIR}")

list(APPEND CMAKE_MODULE_PATH "${CIRCT_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

include(TableGen)
message(STATUS "Using AddCIRCT.cmake in: ${CIRCT_CMAKE_DIR}")
include(AddCIRCT)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(SYSTEM ${CIRCT_INCLUDE_DIRS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})


add_subdirectory(include)
add_subdirectory(lib)

add_subdirectory(cmake)
