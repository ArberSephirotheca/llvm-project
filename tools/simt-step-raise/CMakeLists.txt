add_executable(simt-step-raise
    main.cpp
)

llvm_update_compile_flags(simt-step-raise)

target_link_libraries(simt-step-raise
    PRIVATE
        simt-step
        MLIRIR
        MLIRParser
        MLIRSupport
        MLIRArithDialect
        MLIRFuncDialect
)

mlir_check_link_libraries(simt-step-raise)

install(TARGETS simt-step-raise RUNTIME DESTINATION bin)
