find_package(CUDAToolkit REQUIRED)

add_executable(simt-cuda-test
    main.cpp
)

target_link_libraries(simt-cuda-test
    PRIVATE
    CUDA::cuda_driver
    CUDA::nvrtc
)

target_compile_features(simt-cuda-test PRIVATE cxx_std_20)
