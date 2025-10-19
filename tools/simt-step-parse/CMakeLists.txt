add_executable(simt-step-parse
    src/main.cpp
)

llvm_update_compile_flags(simt-step-parse)

target_link_libraries(simt-step-parse
    PRIVATE
        simt-step
        MLIRIR
        MLIRParser
        MLIRSupport
        MLIRArithDialect
        MLIRFuncDialect
)

mlir_check_link_libraries(simt-step-parse)

install(TARGETS simt-step-parse RUNTIME DESTINATION bin)
