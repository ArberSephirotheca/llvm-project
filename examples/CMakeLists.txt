add_executable(simt-step-run-operation-example
    run_operation_to_buffer.cpp
)

llvm_update_compile_flags(simt-step-run-operation-example)

target_link_libraries(simt-step-run-operation-example
    PRIVATE
        simt-step
        MLIRIR
        MLIRParser
        MLIRSupport
        MLIRArithDialect
        MLIRFuncDialect
)

mlir_check_link_libraries(simt-step-run-operation-example)
