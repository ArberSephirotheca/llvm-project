set(SIMT_OPT_EXECUTABLE "${CMAKE_BINARY_DIR}/tools/simt-opt/simt-opt")

find_program(MLIR_FILE_CHECK
             NAMES mlir-file-check FileCheck
             HINTS "${LLVM_BINARY_DIR}" "${LLVM_BINARY_DIR}/bin")

if(NOT MLIR_FILE_CHECK)
  message(FATAL_ERROR "Unable to locate mlir-file-check/FileCheck")
endif()

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

add_lit_testsuite(check-simt-opt "Running simt-opt regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS simt-opt
  ARGS -j1)

set_target_properties(check-simt-opt PROPERTIES FOLDER "Tests")
