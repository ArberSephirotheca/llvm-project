add_executable(simt-convert
    src/main.cpp
)

set_target_properties(simt-convert PROPERTIES OUTPUT_NAME simt-convert)

target_link_libraries(simt-convert
    PRIVATE
        simt-step
        LLVMSupport
        LLVMOption
)

target_include_directories(simt-convert
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)
