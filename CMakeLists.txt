cmake_minimum_required(VERSION 3.22)

project(SIMTStep LANGUAGES C CXX)

# Allow build to locate LLVM/MLIR from common install prefix if the user
# hasn't provided explicit cache entries. Matches the Rust config defaults.
set(_simt_llvm_prefixes
    "/opt/llvm-hlsl"
)

foreach(_prefix IN LISTS _simt_llvm_prefixes)
    if (NOT LLVM_DIR AND EXISTS "${_prefix}/lib/cmake/llvm")
        set(LLVM_DIR "${_prefix}/lib/cmake/llvm" CACHE PATH "" FORCE)
    endif()
    if (NOT MLIR_DIR AND EXISTS "${_prefix}/lib/cmake/mlir")
        set(MLIR_DIR "${_prefix}/lib/cmake/mlir" CACHE PATH "" FORCE)
    endif()
    if (EXISTS "${_prefix}/lib/cmake")
        list(APPEND CMAKE_PREFIX_PATH "${_prefix}/lib/cmake")
    endif()
endforeach()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(LLVM CONFIG REQUIRED)
find_package(MLIR CONFIG REQUIRED)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using MLIR at ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

if (NOT LLVM_ENABLE_RTTI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

add_subdirectory(lib)
add_subdirectory(tools)
