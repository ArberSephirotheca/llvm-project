find_package(Clang REQUIRED CONFIG)

add_library(simt-frontends-common STATIC
    LoopScopeSupport.cpp
)

target_include_directories(simt-frontends-common
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(simt-frontends-common
    PUBLIC
        simt-step
        MLIRArithDialect
)

if (TARGET clang-cpp)
  target_link_libraries(simt-frontends-common PUBLIC clang-cpp)
else()
  message(FATAL_ERROR "clang-cpp target not found; please ensure the monolithic Clang library is available")
endif()
