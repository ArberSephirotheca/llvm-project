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

add_executable(simple-program-runner Semantics/simple-program-runner.cpp)
target_link_libraries(simple-program-runner PRIVATE
    simt-step
    MLIRParser
    MLIRIR
    MLIRArithDialect
    MLIRFuncDialect
)

add_test(NAME simple_interpreter_if
  COMMAND simple-program-runner ${CMAKE_CURRENT_SOURCE_DIR}/simple-interpreter/if_else.mlir)
set_tests_properties(simple_interpreter_if PROPERTIES PASS_REGULAR_EXPRESSION "1")

add_test(NAME fuzz_supported_ops
  COMMAND /bin/sh -c
          "${CMAKE_BINARY_DIR}/tools/simt-step-fuzz/simt-step-fuzz --lanes=4 --seed=5 --print-ir | grep -E 'arith\\.subi|arith\\.muli' && exit 1 || exit 0")
