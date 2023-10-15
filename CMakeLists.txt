cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(CMAKE_PREFIX_PATH "./resources/libtorch")

project(tglang VERSION 1.0.0 LANGUAGES CXX)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(tglang SHARED tglang.cpp)
#add_executable(tglang tglang.cpp)
target_link_libraries(tglang "${TORCH_LIBRARIES}")
set_property(TARGET tglang PROPERTY CXX_STANDARD 17)