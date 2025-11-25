set(CMAKE_CUDA_ARCHITECTURES 75 80 86) # 75 (Pascal), 80 (Ampere), 86 (Ampere+)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS_DISTRIBUTION "-O3")
set(CMAKE_CXX_STANDARD 17)
enable_language(CUDA)

project(Google_tests
        CXX
        CUDA
)



include(FetchContent)
FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()


# CPU tests
add_executable(cpu_tests
        test_gg_math.cpp
        ${PROJECT_SOURCE_DIR}/../physics.c
)
target_include_directories(cpu_tests PRIVATE ${PROJECT_SOURCE_DIR}/../gglib/src)
if (UNIX)
    target_link_libraries(cpu_tests PRIVATE GTest::gtest_main m gg_math)
else()
    target_link_libraries(cpu_tests PRIVATE GTest::gtest_main gg_math)
endif()
include(GoogleTest)
gtest_discover_tests(cpu_tests)


# GPU tests (optional, only if CUDA available)
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    add_executable(gpu_tests
            test_ggpu_math.cu
            ${PROJECT_SOURCE_DIR}/../ggpulib/gg_math.cu
    )
    set_target_properties(gpu_tests PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES native
    )
    target_include_directories(gpu_tests PRIVATE ${PROJECT_SOURCE_DIR}/../ggpulib)
    target_link_libraries(gpu_tests PRIVATE GTest::gtest_main CUDA::cudart gg_math)
    gtest_discover_tests(gpu_tests)
endif()
