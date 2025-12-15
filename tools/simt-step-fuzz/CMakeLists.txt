add_executable(simt-step-fuzz
    main.cpp
    SimtParticipantPatterns.cpp
    SimtProgramGenerator.cpp
)

llvm_update_compile_flags(simt-step-fuzz)

target_link_libraries(simt-step-fuzz
    PRIVATE
        simt-step
        simt-step-raise-lib
        MLIRIR
        MLIRParser
        MLIRSupport
        MLIRArithDialect
        MLIRFuncDialect
)

mlir_check_link_libraries(simt-step-fuzz)

install(TARGETS simt-step-fuzz RUNTIME DESTINATION bin)
