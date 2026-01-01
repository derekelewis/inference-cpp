cmake_minimum_required(VERSION 3.14)
project(inference-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

set(SOURCES src/main.cpp
    src/safetensors.cpp
    src/safetensors.h
    src/tensor.cpp
    src/tensor.h
)

add_executable(inference-cpp
    src/main.cpp
    src/safetensors.cpp
    src/tensor.cpp
)

target_compile_options(inference-cpp PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(inference-cpp PRIVATE nlohmann_json::nlohmann_json)
target_include_directories(inference-cpp PRIVATE src)