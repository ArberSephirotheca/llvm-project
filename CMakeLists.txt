cmake_minimum_required(VERSION 3.22)

if (NOT CMAKE_C_COMPILER)
    find_program(_clang_exe NAMES clang)
    if (_clang_exe)
        set(CMAKE_C_COMPILER "${_clang_exe}" CACHE FILEPATH "" FORCE)
    endif()
endif()

if (NOT CMAKE_CXX_COMPILER)
    find_program(_clangxx_exe NAMES clang++)
    if (_clangxx_exe)
        set(CMAKE_CXX_COMPILER "${_clangxx_exe}" CACHE FILEPATH "" FORCE)
    endif()
endif()

project(SIMTStep LANGUAGES C CXX)

# Allow build to locate LLVM/MLIR from common install prefixes if the user
# hasn't provided explicit cache entries. Matches the Rust config defaults.
set(_simt_llvm_prefixes)
if (DEFINED ENV{LLVM_PREFIX})
    list(APPEND _simt_llvm_prefixes "$ENV{LLVM_PREFIX}")
endif()
list(APPEND _simt_llvm_prefixes "/opt/llvm")

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

enable_testing()

if (NOT LLVM_ENABLE_RTTI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

find_program(CLANG_FORMAT_EXE NAMES clang-format)
if (CLANG_FORMAT_EXE)
    set(_format_search_dirs
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/lib
        ${PROJECT_SOURCE_DIR}/tools
        ${PROJECT_SOURCE_DIR}/test
        ${PROJECT_SOURCE_DIR}/docs
    )
    set(_format_extensions h hh hpp hxx c cc cpp cxx cu)
    set(FORMAT_FILES)
    foreach(_dir IN LISTS _format_search_dirs)
        if (EXISTS "${_dir}")
            foreach(_ext IN LISTS _format_extensions)
                file(GLOB_RECURSE _format_tmp CONFIGURE_DEPENDS
                    "${_dir}/*.${_ext}")
                list(APPEND FORMAT_FILES ${_format_tmp})
            endforeach()
        endif()
    endforeach()
    list(REMOVE_DUPLICATES FORMAT_FILES)
    if (FORMAT_FILES)
        add_custom_target(format
            COMMAND ${CMAKE_COMMAND}
                -DCLANG_FORMAT=${CLANG_FORMAT_EXE}
                -DFILES=${FORMAT_FILES}
                -P ${PROJECT_SOURCE_DIR}/cmake/run_clang_format.cmake
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            COMMENT "Running clang-format on project sources"
        )
    else()
        message(STATUS "No source files found for clang-format target")
    endif()
else()
    message(STATUS "clang-format not found; 'format' target disabled")
endif()

add_subdirectory(lib)
add_subdirectory(tools)
