cmake_minimum_required(VERSION 4.1)
project(inference-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SOURCES src/main.cpp
    src/safetensors.cpp
    src/safetensors.h
)

add_executable(inference-cpp
    src/main.cpp
    src/safetensors.cpp
)

target_compile_options(inference-cpp PRIVATE -Wall -Wextra -Wpedantic)

target_include_directories(inference-cpp PRIVATE src)