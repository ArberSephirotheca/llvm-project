set(LLVM_TARGET_DEFINITIONS
    ${PROJECT_SOURCE_DIR}/include/simt-step/Dialect/SimtStructured/Structured.td)

set(MLIR_TABLEGEN_INCLUDE_DIRECTORIES
    ${MLIR_MAIN_INCLUDE_DIR}
    ${LLVM_MAIN_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/include)

set(_simt_struct_tablegen_includes
    -I${MLIR_MAIN_INCLUDE_DIR}
    -I${LLVM_MAIN_INCLUDE_DIR}
    -I${PROJECT_SOURCE_DIR}/include
    -I/opt/llvm-20/include)

mlir_tablegen(StructuredDialect.h.inc -gen-dialect-decls -dialect=simt_struct
    ${_simt_struct_tablegen_includes})
mlir_tablegen(StructuredDialect.cpp.inc -gen-dialect-defs -dialect=simt_struct
    ${_simt_struct_tablegen_includes})
mlir_tablegen(StructuredOps.h.inc -gen-op-decls
    ${_simt_struct_tablegen_includes})
mlir_tablegen(StructuredOps.cpp.inc -gen-op-defs
    ${_simt_struct_tablegen_includes})
mlir_tablegen(StructuredEnums.h.inc -gen-enum-decls
    ${_simt_struct_tablegen_includes})
mlir_tablegen(StructuredEnums.cpp.inc -gen-enum-defs
    ${_simt_struct_tablegen_includes})

add_public_tablegen_target(SimtStructuredDialectIncGen)

set(SIMT_STRUCTURED_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/StructuredDialect.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/StructuredOps.cpp)

foreach(src ${SIMT_STRUCTURED_SOURCES})
  target_sources(simt-step PRIVATE ${src})
endforeach()

target_include_directories(simt-step
    PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(simt-step
    PRIVATE
        MLIRIR
        MLIRControlFlowInterfaces
)

add_dependencies(simt-step SimtStructuredDialectIncGen)
