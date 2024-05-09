set(DFCIR_PACKAGE_DIR lib/cmake/DFCIR)

set(CONF_DIR ${CMAKE_BINARY_DIR}/${DFCIR_PACKAGE_DIR})

export(TARGETS MLIRDFCIR
       NAMESPACE DFCIR::
       FILE ${CONF_DIR}/DFCIRTargets.cmake
       )

configure_file(
        ${CMAKE_CURRENT_LIST_DIR}/FindLpSolve.cmake
        ${CONF_DIR}/FindLpSolve.cmake
        COPYONLY
        )

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/DFCIRConfig.cmake.in
        ${CONF_DIR}/DFCIRConfig.cmake
        @ONLY
        )