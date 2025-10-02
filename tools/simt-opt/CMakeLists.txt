add_executable(simt-opt
    src/main.cpp
)

llvm_update_compile_flags(simt-opt)

target_link_libraries(simt-opt
    PRIVATE
        simt-step
        SimtStepToStructuredPass
        MLIRGPUToLLVMSPV
        MLIRGPUToSPIRV
        MLIRIR
        MLIRControlFlowDialect
        MLIRSupport
        MLIRParser
        MLIROptLib
)

mlir_check_link_libraries(simt-opt)

install(TARGETS simt-opt RUNTIME DESTINATION bin)
