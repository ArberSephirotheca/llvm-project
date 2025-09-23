find_package(Clang REQUIRED CONFIG)

add_executable(simt-hlsl-import
    src/main.cpp
)

set_target_properties(simt-hlsl-import PROPERTIES OUTPUT_NAME simt-hlsl-import)

llvm_update_compile_flags(simt-hlsl-import)

if (TARGET clang-cpp)
  target_link_libraries(simt-hlsl-import PRIVATE clang-cpp)
else()
  message(FATAL_ERROR "clang-cpp target not found; please ensure the monolithic Clang library is available")
endif()

target_link_libraries(simt-hlsl-import
    PRIVATE
        simt-step
        MLIRSupport
        MLIRFuncDialect
        MLIRArithDialect
        LLVMCore
        clangTooling
)

target_include_directories(simt-hlsl-import
    PRIVATE
        ${CLANG_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
)
