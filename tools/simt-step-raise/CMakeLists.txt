add_library(simt-step-raise-lib
    HlslEmitter.cpp
)

target_include_directories(simt-step-raise-lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

llvm_update_compile_flags(simt-step-raise-lib)

target_link_libraries(simt-step-raise-lib
    PUBLIC
        simt-step
        MLIRIR
        MLIRParser
        MLIRSupport
        MLIRArithDialect
        MLIRFuncDialect
)

add_executable(simt-step-raise
    main.cpp
)

llvm_update_compile_flags(simt-step-raise)

target_link_libraries(simt-step-raise
    PRIVATE
        simt-step-raise-lib
)

mlir_check_link_libraries(simt-step-raise)

install(TARGETS simt-step-raise RUNTIME DESTINATION bin)
