add_executable(simt-run
    src/main.cpp
)

set_target_properties(simt-run PROPERTIES OUTPUT_NAME simt-run)

target_link_libraries(simt-run
    PRIVATE
        simt-step
)

target_include_directories(simt-run
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)
