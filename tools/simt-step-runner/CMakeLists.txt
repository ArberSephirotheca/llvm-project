add_executable(simt-step-runner
    main.cpp
)

llvm_update_compile_flags(simt-step-runner)

target_link_libraries(simt-step-runner
    PRIVATE
        simt-step
        MLIRIR
        MLIRParser
        MLIRSupport
        MLIRArithDialect
        MLIRFuncDialect
)

mlir_check_link_libraries(simt-step-runner)

install(TARGETS simt-step-runner RUNTIME DESTINATION bin)
