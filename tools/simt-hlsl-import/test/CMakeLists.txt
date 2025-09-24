set(SIMT_HLSL_IMPORT_TEST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SIMT_HLSL_IMPORT_TEST_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(SIMT_HLSL_IMPORT_TEST_EXECUTABLE
    "${CMAKE_BINARY_DIR}/tools/simt-hlsl-import/simt-hlsl-import")

# Resolve FileCheck from the in-tree LLVM build.
find_program(MLIR_FILE_CHECK
             NAMES mlir-file-check FileCheck
             HINTS "${LLVM_BINARY_DIR}" "${LLVM_BINARY_DIR}/bin")

if(NOT EXISTS "${SIMT_HLSL_IMPORT_TEST_EXECUTABLE}")
  message(FATAL_ERROR "simt-hlsl-import binary not found: "
                      "${SIMT_HLSL_IMPORT_TEST_EXECUTABLE}")
endif()

if(NOT MLIR_FILE_CHECK)
  message(FATAL_ERROR "Unable to locate mlir-file-check/FileCheck")
endif()

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

add_lit_testsuite(check-simt-hlsl-import "Running simt-hlsl-import regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS simt-hlsl-import
  ARGS -j1)

set_target_properties(check-simt-hlsl-import PROPERTIES FOLDER "Tests")


