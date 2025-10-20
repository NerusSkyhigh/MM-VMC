cmake_minimum_required(VERSION 4.0)
project(QMC C)

set(CMAKE_C_STANDARD 11)
SET(CMAKE_C_FLAGS_DISTRIBUTION "-O3")
SET(CMAKE_C_FLAGS "-pedantic -Wall -Wextra -Wconversion")

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
