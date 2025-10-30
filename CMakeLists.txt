cmake_minimum_required(VERSION 3.10)
project(QMC
        C
        CUDA
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS_DISTRIBUTION "-O3")
if (UNIX)
    set(CMAKE_C_FLAGS "-pedantic -Wall -Wextra -Wconversion")
endif()



add_executable(QMC DiffusionMonteCarlo.c
        gglib/gg_math.h
        gglib/gg_math.c
        gglib/gg_mem.c
        physics.c
        physics.h
)

add_executable(VMC VariationalMonteCarlo.c
        gglib/gg_math.h
        gglib/gg_math.c
        gglib/gg_mem.c
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
        physics.cuh
)

add_subdirectory(tests)