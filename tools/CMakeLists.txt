add_subdirectory(simt-hlsl-import)
add_subdirectory(simt-step-fuzz)
add_subdirectory(simt-step-raise)
add_subdirectory(simt-step-runner)

option(SIMT_STEP_BUILD_CUDA_TEST "Build simt-cuda-test (requires CUDA)" ON)
if (SIMT_STEP_BUILD_CUDA_TEST)
    add_subdirectory(simt-cuda-test)
endif()
