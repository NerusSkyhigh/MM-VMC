cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86) # 75 (Pascal), 80 (Ampere), 86 (Ampere+)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS_DISTRIBUTION "-O3")
set(CMAKE_CXX_STANDARD 17)

project(QMC
        C
        CUDA
)
enable_language(CUDA)

if (UNIX)
    set(CMAKE_C_FLAGS "-pedantic -Wall -Wextra -Wconversion")
endif()

add_subdirectory(gglib-c)

add_executable(QMC DiffusionMonteCarlo.c
        gglib-c/include/gg_math.h
        gglib-c/src/gg_math.c
        physics.c
        physics.h
)

add_executable(VMC VariationalMonteCarlo.c
        gglib-c/include/gg_math.h
        gglib-c/src/gg_math.c
        physics.c
        physics.h
)

target_link_libraries(VMC
        m
)

add_executable(gpuVMC VariationalMonteCarlo.cu
        ggpulib/gg_math.cuh
        ggpulib/gg_math.cu
        ggpulib/gg_mem.cu
        ggpulib/gg_mem.cuh
        ggpulib/gg_cuda_macro.cuh

        gglib-c/include/gg_math.h
        gglib-c/src/gg_math.c
        physics.cuh
        gglib-c/src/gg_io.c
        gglib-c/include/gg_io.h
)
set_target_properties(gpuVMC PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
)


add_subdirectory(tests)