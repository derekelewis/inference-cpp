cmake_minimum_required(VERSION 3.14)
project(inference-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# JSON library
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

# ICU library (system-installed)
find_package(ICU REQUIRED COMPONENTS uc i18n)

# Google Test
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(SOURCES src/main.cpp
    src/safetensors.cpp
    src/safetensors.h
    src/tensor.cpp
    src/tensor.h
)

add_executable(inference-cpp
    src/main.cpp
    src/qwen3.cpp
    src/safetensors.cpp
    src/tensor.cpp
    src/tokenizer.cpp
)

target_compile_options(inference-cpp PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(inference-cpp PRIVATE
    nlohmann_json::nlohmann_json
    ICU::uc
    ICU::i18n
)
target_include_directories(inference-cpp PRIVATE src)

# Tests
enable_testing()

add_executable(tensor_tests
    tests/tensor_test.cpp
    src/tensor.cpp
)

target_compile_options(tensor_tests PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(tensor_tests PRIVATE src)
target_link_libraries(tensor_tests PRIVATE GTest::gtest_main)

add_executable(safetensor_tests
    tests/safetensor_test.cpp
    src/safetensors.cpp
    src/tensor.cpp
)

target_compile_options(safetensor_tests PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(safetensor_tests PRIVATE src)
target_link_libraries(safetensor_tests PRIVATE
    GTest::gtest_main
    nlohmann_json::nlohmann_json
)

add_executable(qwen3_tests
    tests/qwen3_test.cpp
    src/qwen3.cpp
    src/safetensors.cpp
    src/tensor.cpp
)

target_compile_options(qwen3_tests PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(qwen3_tests PRIVATE src)
target_link_libraries(qwen3_tests PRIVATE
    GTest::gtest_main
    nlohmann_json::nlohmann_json
)

# Forward pass tests (slow, requires model, run separately)
add_executable(qwen3_forward_tests
    tests/qwen3_forward_test.cpp
    src/qwen3.cpp
    src/safetensors.cpp
    src/tensor.cpp
)

target_compile_options(qwen3_forward_tests PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(qwen3_forward_tests PRIVATE src)
target_link_libraries(qwen3_forward_tests PRIVATE
    GTest::gtest_main
    nlohmann_json::nlohmann_json
)

add_executable(tokenizer_tests
    tests/tokenizer_test.cpp
    src/tokenizer.cpp
)

target_compile_options(tokenizer_tests PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(tokenizer_tests PRIVATE src)
target_link_libraries(tokenizer_tests PRIVATE
    GTest::gtest_main
    nlohmann_json::nlohmann_json
    ICU::uc
    ICU::i18n
)

include(GoogleTest)
gtest_discover_tests(tensor_tests)
gtest_discover_tests(safetensor_tests)
# Run qwen3_tests as single test to share model loading across tests
add_test(NAME qwen3_tests COMMAND qwen3_tests)
# Run tokenizer_tests as single test to share tokenizer loading across tests
add_test(NAME tokenizer_tests COMMAND tokenizer_tests)

# Forward pass tests - slow, requires model weights
# Usage:
#   ctest -E forward      # Run all tests EXCEPT forward tests (fast)
#   ctest -L slow         # Run ONLY slow tests (forward tests)
#   ./qwen3_forward_tests # Run forward tests directly
add_test(NAME qwen3_forward_tests COMMAND qwen3_forward_tests)
set_tests_properties(qwen3_forward_tests PROPERTIES LABELS "slow")
