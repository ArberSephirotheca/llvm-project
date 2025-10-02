add_library(simt-step)

target_include_directories(simt-step
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(simt-step
    PUBLIC
        MLIRIR
        MLIRSupport
        LLVMSupport
)

add_subdirectory(core)
add_subdirectory(plugins)
add_subdirectory(semantics)
add_subdirectory(frontends)
add_subdirectory(Dialect)
add_subdirectory(Conversion)
