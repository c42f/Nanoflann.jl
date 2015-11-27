set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")

project(Nanoflann)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-Wall")

include_directories(src/include)
add_library(nanoflann SHARED "nanoflann.cpp")

get_target_property(libpath nanoflann LOCATION)

file(WRITE ${CMAKE_BINARY_DIR}/deps.jl "const nanoflann_lib = \"${libpath}\"\n")
