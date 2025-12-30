# inference-cpp

## Overview

`inference-cpp` is a project to implement a basic inferencing framework written in C++ (C++11). The goal is to be able to load supported models and their respective layers from one more or more .safetensors files associated with the model, tokenize input, perform a forward pass, and decode the output tokens. 

The target model to be implemented is Qwen2 -- more specifically, the `Qwen/Qwen3-4B-Instruct-2507` model.

## Status

1. Model layer loading into tensors (In Progress)
2. `Qwen2Tokernizer` encode/decode (Not Started)
3. `Tensor` and `Shape` class implementation (In Progress)
4. Unit tests (Not Started)
5. bloat16 implementation (Not Started)

## Assumptions

1. Treat .safetensor files as trusted inputs
2. CPU decoding planned initially with plans to support CUDA and Metal after a forward pass with CPU can be completed.