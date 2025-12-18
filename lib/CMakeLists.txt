add_library(simt-step)

target_include_directories(simt-step
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(simt-step
    PUBLIC
        MLIRIR
        MLIRSupport
        LLVMSupport
        MLIRPass
        MLIRArithDialect
        MLIRFuncDialect
)

add_subdirectory(core)
add_subdirectory(semantics)
add_subdirectory(frontends)
add_subdirectory(Dialect)
