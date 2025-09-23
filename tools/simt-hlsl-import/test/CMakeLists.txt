set(SIMT_HLSL_IMPORT_TEST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SIMT_HLSL_IMPORT_TEST_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

add_lit_testsuite(check-simt-hlsl-import "Running simt-hlsl-import regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS simt-hlsl-import
  ARGS -j1)

set_target_properties(check-simt-hlsl-import PROPERTIES FOLDER "Tests")
