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
)

add_executable(VMC VariationalMonteCarlo.c
        gglib/gg_math.h
        gglib/gg_math.c
        gglib/gg_mem.c
        physics.c
)

target_link_libraries(VMC
        m
)

add_subdirectory(tests)