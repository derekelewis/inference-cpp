# inference-cpp

## Overview

`inference-cpp` is a minimal LLM inference engine written in C++11. It loads models from safetensor files, tokenizes input, performs forward passes, and generates text.

Currently supports the `Qwen/Qwen3-4B-Instruct-2507` model.

## Building

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

## Usage

```bash
./inference-cpp <model_path> [prompt] [max_tokens]

# Example
./inference-cpp ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/<hash> "Hello, I am" 20
```

## Running Tests

Tests use Google Test framework (automatically fetched during CMake configuration).

```bash
cd build

# Fast tests only (~13 seconds, excludes slow forward pass tests)
make -j4 && ctest -E forward --output-on-failure

# All tests including forward pass (~3+ minutes, requires model)
ctest --output-on-failure

# Forward pass tests only
ctest -L slow --output-on-failure
# OR
./qwen3_forward_tests
```

## Status

| Component | Status |
|-----------|--------|
| Safetensor loading | Complete |
| BPE Tokenizer (encode/decode) | Complete |
| Tensor operations | Complete |
| RoPE (Rotary Position Embeddings) | Complete |
| Grouped-Query Attention (GQA) | Complete |
| KV Cache | Complete |
| Forward pass | Complete |
| Token generation | Complete |
| Unit tests | 160 tests |

## Architecture

- **Tensor**: N-dimensional array with broadcasting, matmul, softmax, RMS norm, RoPE
- **Tokenizer**: BPE tokenizer with Unicode normalization (ICU)
- **Qwen3Model**: Transformer with GQA (32 Q heads, 8 KV heads), SiLU MLP, Q/K norms

## Performance

Current performance on CPU (single-threaded, no SIMD):
- ~2.7 seconds per token (Qwen3-4B, Release build)

Uses stride-based tensor views to minimize memory copies. See `docs/tensor-views-migration.md` for implementation details.

## Assumptions

1. Safetensor files are treated as trusted inputs
2. CPU inference only (CUDA/Metal planned for future)