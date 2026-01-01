# inference-cpp

## Overview

`inference-cpp` is a minimal LLM inference engine written in C++11. It loads models from safetensor files, tokenizes input, performs forward passes, and generates text.

Currently supports the `Qwen/Qwen3-4B-Instruct-2507` model.

## Building

Requires: CMake 3.14+, C++11 compiler, ICU library, OpenMP

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

## Usage

```bash
./inference-cpp <model_path> [options]

Options:
  -p, --prompt <text>    Text prompt (default: "Hello, I am")
  -n, --max-tokens <N>   Maximum tokens to generate (default: 20)
  -t, --threads <N>      Number of threads (default: all cores)
  -h, --help             Show help message

# Examples
./inference-cpp ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/<hash>
./inference-cpp <model_path> -t 8 -p "What is AI?" -n 50
./inference-cpp <model_path> --threads 4 --prompt "Hello" --max-tokens 30

# Legacy positional arguments also supported for backward compatibility
./inference-cpp <model_path> "Hello, I am" 20
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
| OpenMP multithreading | Complete |
| Unit tests | 159 tests |

## Architecture

- **Tensor**: N-dimensional array with broadcasting, matmul, softmax, RMS norm, RoPE
- **Tokenizer**: BPE tokenizer with Unicode normalization (ICU)
- **Qwen3Model**: Transformer with GQA (32 Q heads, 8 KV heads), SiLU MLP, Q/K norms

## Performance

Current performance on CPU (Qwen3-4B, Release build, no SIMD):
- Single-threaded: ~0.37 tok/s (~2.7s per token)
- Multi-threaded (20 threads): ~4.3 tok/s (~11x speedup)

Use `-t N` to set thread count (defaults to all cores).

OpenMP parallelization targets:
- Matrix multiplications (2D, 3D, 4D with GQA support)
- Attention Q/K normalization
- Causal mask application

Uses stride-based tensor views to minimize memory copies. See `docs/tensor-views-migration.md` for implementation details.

## Assumptions

1. Safetensor files are treated as trusted inputs
2. CPU inference only (CUDA/Metal planned for future)