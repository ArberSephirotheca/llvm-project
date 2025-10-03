add_library(SimtStepToStructuredPass STATIC
    SimtStepToStructured.cpp
    StructuredCFGBuilder.cpp
)

target_include_directories(SimtStepToStructuredPass
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(SimtStepToStructuredPass
    PRIVATE
        simt-step
        MLIRIR
        MLIRFuncDialect
        MLIRSupport
    PUBLIC
        MLIRControlFlowDialect
)
