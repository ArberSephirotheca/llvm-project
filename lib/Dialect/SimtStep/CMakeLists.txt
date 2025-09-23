set(LLVM_TARGET_DEFINITIONS
    ${PROJECT_SOURCE_DIR}/include/simt-step/Dialect/SimtStep/SimtStep.td)

set(MLIR_TABLEGEN_INCLUDE_DIRECTORIES
    ${MLIR_MAIN_INCLUDE_DIR}
    ${LLVM_MAIN_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/include)

set(_simt_tablegen_includes
    -I${MLIR_MAIN_INCLUDE_DIR}
    -I${LLVM_MAIN_INCLUDE_DIR}
    -I${PROJECT_SOURCE_DIR}/include
    -I/opt/llvm-hlsl/include)

mlir_tablegen(SimtStepDialect.h.inc -gen-dialect-decls -dialect=simt_step
    ${_simt_tablegen_includes})
mlir_tablegen(SimtStepDialect.cpp.inc -gen-dialect-defs -dialect=simt_step
    ${_simt_tablegen_includes})
mlir_tablegen(SimtStepOps.h.inc -gen-op-decls
    ${_simt_tablegen_includes})
mlir_tablegen(SimtStepOps.cpp.inc -gen-op-defs
    ${_simt_tablegen_includes})
mlir_tablegen(SimtStepEnums.h.inc -gen-enum-decls
    ${_simt_tablegen_includes})
mlir_tablegen(SimtStepEnums.cpp.inc -gen-enum-defs
    ${_simt_tablegen_includes})
mlir_tablegen(SimtStepTypes.h.inc -gen-typedef-decls
    ${_simt_tablegen_includes})
mlir_tablegen(SimtStepTypes.cpp.inc -gen-typedef-defs
    ${_simt_tablegen_includes})

add_public_tablegen_target(SimtStepDialectIncGen)

set(SIMT_STEP_DIALECT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/SimtStepDialect.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SimtStepOps.cpp)

foreach(src ${SIMT_STEP_DIALECT_SOURCES})
  target_sources(simt-step PRIVATE ${src})
endforeach()

target_include_directories(simt-step
    PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
)

add_dependencies(simt-step SimtStepDialectIncGen)
