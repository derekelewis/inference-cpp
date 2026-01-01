#pragma once

#include <string>
#include <vector>

#include "tensor.h"

// ============================================================================
// Qwen3 Model Configuration
// ============================================================================
struct Qwen3Config {
  size_t vocab_size = 151936;
  size_t hidden_size = 2560;
  size_t intermediate_size = 9728;
  size_t num_hidden_layers = 36;
  size_t num_attention_heads = 32;
  size_t num_key_value_heads = 8;
  size_t head_dim = 128;
  float rms_norm_eps = 1e-6f;
};

// ============================================================================
// Qwen3 Attention Weights
// ============================================================================
struct Qwen3Attention {
  Tensor<float> q_proj;  // [num_heads * head_dim, hidden_size]
  Tensor<float> k_proj;  // [num_kv_heads * head_dim, hidden_size]
  Tensor<float> v_proj;  // [num_kv_heads * head_dim, hidden_size]
  Tensor<float> o_proj;  // [hidden_size, num_heads * head_dim]
  Tensor<float> q_norm;  // [head_dim]
  Tensor<float> k_norm;  // [head_dim]
};

// ============================================================================
// Qwen3 MLP Weights
// ============================================================================
struct Qwen3MLP {
  Tensor<float> gate_proj;  // [intermediate_size, hidden_size]
  Tensor<float> up_proj;    // [intermediate_size, hidden_size]
  Tensor<float> down_proj;  // [hidden_size, intermediate_size]
};

// ============================================================================
// Qwen3 Transformer Layer
// ============================================================================
struct Qwen3Layer {
  Tensor<float> input_layernorm;          // [hidden_size]
  Qwen3Attention self_attn;
  Tensor<float> post_attention_layernorm;  // [hidden_size]
  Qwen3MLP mlp;
};

// ============================================================================
// Qwen3 Model
// ============================================================================
class Qwen3Model {
 public:
  // Load model weights from a directory containing safetensor files
  explicit Qwen3Model(const std::string& model_path);

  // Accessors
  const Qwen3Config& config() const { return config_; }
  const Tensor<float>& embed_tokens() const { return embed_tokens_; }
  const Tensor<float>& norm() const { return norm_; }
  const Qwen3Layer& layer(size_t idx) const;
  size_t num_layers() const { return layers_.size(); }

 private:
  Qwen3Config config_;
  Tensor<float> embed_tokens_;       // [vocab_size, hidden_size]
  std::vector<Qwen3Layer> layers_;   // num_hidden_layers layers
  Tensor<float> norm_;               // [hidden_size] - final RMS norm
};
