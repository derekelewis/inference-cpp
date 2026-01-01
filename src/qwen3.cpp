#include "qwen3.h"

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "safetensors.h"

Qwen3Model::Qwen3Model(const std::string& model_path) {
  ModelLoader loader(model_path);

  // Load embedding weights
  embed_tokens_ = loader.load_tensor("model.embed_tokens.weight");

  // Load all transformer layers
  layers_.reserve(config_.num_hidden_layers);

  for (size_t i = 0; i < config_.num_hidden_layers; ++i) {
    std::string prefix = "model.layers." + std::to_string(i) + ".";

    Qwen3Layer layer;

    // Layer norms
    layer.input_layernorm =
        loader.load_tensor(prefix + "input_layernorm.weight");
    layer.post_attention_layernorm =
        loader.load_tensor(prefix + "post_attention_layernorm.weight");

    // Attention projections
    layer.self_attn.q_proj =
        loader.load_tensor(prefix + "self_attn.q_proj.weight");
    layer.self_attn.k_proj =
        loader.load_tensor(prefix + "self_attn.k_proj.weight");
    layer.self_attn.v_proj =
        loader.load_tensor(prefix + "self_attn.v_proj.weight");
    layer.self_attn.o_proj =
        loader.load_tensor(prefix + "self_attn.o_proj.weight");

    // Query/Key norms (Qwen3 specific)
    layer.self_attn.q_norm =
        loader.load_tensor(prefix + "self_attn.q_norm.weight");
    layer.self_attn.k_norm =
        loader.load_tensor(prefix + "self_attn.k_norm.weight");

    // MLP projections
    layer.mlp.gate_proj = loader.load_tensor(prefix + "mlp.gate_proj.weight");
    layer.mlp.up_proj = loader.load_tensor(prefix + "mlp.up_proj.weight");
    layer.mlp.down_proj = loader.load_tensor(prefix + "mlp.down_proj.weight");

    layers_.push_back(std::move(layer));
  }

  // Load final norm
  norm_ = loader.load_tensor("model.norm.weight");
}

const Qwen3Layer& Qwen3Model::layer(size_t idx) const {
  if (idx >= layers_.size()) {
    throw std::out_of_range("Layer index out of range");
  }
  return layers_[idx];
}

// ============================================================================
// Embedding lookup
// ============================================================================
Tensor<float> Qwen3Model::embed(const std::vector<uint32_t>& tokens) const {
  size_t seq_len = tokens.size();
  size_t hidden_size = config_.hidden_size;

  // Output: [1, seq_len, hidden_size] (batch=1)
  Tensor<float> embeddings(Shape({1, seq_len, hidden_size}));

  for (size_t i = 0; i < seq_len; ++i) {
    uint32_t token_id = tokens[i];
    if (token_id >= config_.vocab_size) {
      throw std::out_of_range("Token ID out of vocabulary range");
    }

    // Copy embedding vector for this token
    for (size_t j = 0; j < hidden_size; ++j) {
      embeddings(0, i, j) = embed_tokens_(token_id, j);
    }
  }

  return embeddings;
}

// ============================================================================
// MLP: gate_proj * silu(up_proj) -> down_proj
// ============================================================================
Tensor<float> Qwen3Model::mlp(const Tensor<float>& hidden_states,
                               const Qwen3MLP& mlp_weights) const {
  // hidden_states: [batch, seq_len, hidden_size]
  // gate_proj: [intermediate_size, hidden_size]
  // up_proj: [intermediate_size, hidden_size]
  // down_proj: [hidden_size, intermediate_size]

  size_t batch = hidden_states.size(0);
  size_t seq_len = hidden_states.size(1);
  size_t hidden_size = hidden_states.size(2);

  // Reshape to 2D for matmul: [batch * seq_len, hidden_size]
  Tensor<float> x = hidden_states.reshape(Shape({batch * seq_len, hidden_size}));

  // gate = x @ gate_proj.T -> [batch * seq_len, intermediate_size]
  Tensor<float> gate = x.matmul(mlp_weights.gate_proj.transpose());

  // up = x @ up_proj.T -> [batch * seq_len, intermediate_size]
  Tensor<float> up = x.matmul(mlp_weights.up_proj.transpose());

  // Apply SiLU to gate and multiply with up
  gate = gate.silu();
  Tensor<float> gated = gate * up;

  // down = gated @ down_proj.T -> [batch * seq_len, hidden_size]
  Tensor<float> output = gated.matmul(mlp_weights.down_proj.transpose());

  // Reshape back to [batch, seq_len, hidden_size]
  return output.reshape(Shape({batch, seq_len, hidden_size}));
}

// ============================================================================
// Attention with GQA support
// ============================================================================
Tensor<float> Qwen3Model::attention(const Tensor<float>& hidden_states,
                                     const Qwen3Attention& attn,
                                     size_t layer_idx,
                                     size_t start_pos,
                                     KVCache* cache) const {
  // hidden_states: [batch, seq_len, hidden_size]
  size_t batch = hidden_states.size(0);
  size_t seq_len = hidden_states.size(1);
  size_t hidden_size = hidden_states.size(2);
  size_t num_heads = config_.num_attention_heads;
  size_t num_kv_heads = config_.num_key_value_heads;
  size_t head_dim = config_.head_dim;

  // Reshape to 2D for projection: [batch * seq_len, hidden_size]
  Tensor<float> x = hidden_states.reshape(Shape({batch * seq_len, hidden_size}));

  // Q, K, V projections
  // q_proj: [num_heads * head_dim, hidden_size]
  // k_proj: [num_kv_heads * head_dim, hidden_size]
  // v_proj: [num_kv_heads * head_dim, hidden_size]
  Tensor<float> q = x.matmul(attn.q_proj.transpose());  // [batch*seq, num_heads*head_dim]
  Tensor<float> k = x.matmul(attn.k_proj.transpose());  // [batch*seq, num_kv_heads*head_dim]
  Tensor<float> v = x.matmul(attn.v_proj.transpose());  // [batch*seq, num_kv_heads*head_dim]

  // Reshape to [batch, seq_len, num_heads, head_dim]
  q = q.reshape(Shape({batch, seq_len, num_heads, head_dim}));
  k = k.reshape(Shape({batch, seq_len, num_kv_heads, head_dim}));
  v = v.reshape(Shape({batch, seq_len, num_kv_heads, head_dim}));

  // Apply Q/K normalization BEFORE RoPE (Qwen3 specific)
  // q_norm and k_norm are [head_dim] - apply per-head RMS norm

  // Q norm: parallelize over batch, seq, and head dimensions
#pragma omp parallel for collapse(3) schedule(static)
  for (size_t b = 0; b < batch; ++b) {
    for (size_t s = 0; s < seq_len; ++s) {
      for (size_t h = 0; h < num_heads; ++h) {
        float sum_sq = 0.0f;
        for (size_t d = 0; d < head_dim; ++d) {
          float val = q(b, s, h, d);
          sum_sq += val * val;
        }
        float rms = std::sqrt(sum_sq / static_cast<float>(head_dim) + config_.rms_norm_eps);
        for (size_t d = 0; d < head_dim; ++d) {
          q(b, s, h, d) = (q(b, s, h, d) / rms) * attn.q_norm(d);
        }
      }
    }
  }

  // K norm: parallelize over batch, seq, and kv_head dimensions
#pragma omp parallel for collapse(3) schedule(static)
  for (size_t b = 0; b < batch; ++b) {
    for (size_t s = 0; s < seq_len; ++s) {
      for (size_t h = 0; h < num_kv_heads; ++h) {
        float sum_sq = 0.0f;
        for (size_t d = 0; d < head_dim; ++d) {
          float val = k(b, s, h, d);
          sum_sq += val * val;
        }
        float rms = std::sqrt(sum_sq / static_cast<float>(head_dim) + config_.rms_norm_eps);
        for (size_t d = 0; d < head_dim; ++d) {
          k(b, s, h, d) = (k(b, s, h, d) / rms) * attn.k_norm(d);
        }
      }
    }
  }

  // Transpose to [batch, num_heads, seq_len, head_dim] for RoPE
  // RoPE expects [..., seq_len, head_dim] where seq_len is the second-to-last dim
  q = q.transpose(1, 2);  // [batch, num_heads, seq_len, head_dim]
  k = k.transpose(1, 2);  // [batch, num_kv_heads, seq_len, head_dim]
  v = v.transpose(1, 2);  // [batch, num_kv_heads, seq_len, head_dim]

  // Apply RoPE to Q and K (now seq_len is correctly in dim -2)
  q = q.rope(start_pos);
  k = k.rope(start_pos);

  // Handle KV cache
  // K, V are already [batch, num_kv_heads, seq_len, head_dim]
  size_t total_len = seq_len;
  if (cache) {
    if (layer_idx < cache->k_cache.size()) {
      // Concatenate with existing cache along seq dimension (dim 2)
      std::vector<Tensor<float>> k_list = {cache->k_cache[layer_idx], k};
      std::vector<Tensor<float>> v_list = {cache->v_cache[layer_idx], v};
      k = Tensor<float>::concat(k_list, 2);
      v = Tensor<float>::concat(v_list, 2);

      // Update cache
      cache->k_cache[layer_idx] = k;
      cache->v_cache[layer_idx] = v;
      total_len = k.size(2);
    } else {
      // First time: store in cache
      cache->k_cache.push_back(k);
      cache->v_cache.push_back(v);
      cache->seq_len = seq_len;
    }
  }

  // Q is already [batch, num_heads, seq_len, head_dim]

  // Compute attention scores: Q @ K^T
  // Q: [batch, num_heads, seq_len, head_dim]
  // K: [batch, num_kv_heads, total_len, head_dim]
  // K^T: [batch, num_kv_heads, head_dim, total_len]
  Tensor<float> k_t = k.transpose(2, 3);

  // scores = Q @ K^T -> [batch, num_heads, seq_len, total_len]
  // GQA broadcasting is handled by matmul
  Tensor<float> scores = q.matmul(k_t);

  // Scale by sqrt(head_dim)
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  scores = scores * scale;

  // Apply causal mask
  Tensor<float> mask = Tensor<float>::causal_mask(seq_len, total_len);
  // Broadcast mask to scores shape
#pragma omp parallel for collapse(4) schedule(static)
  for (size_t b = 0; b < batch; ++b) {
    for (size_t h = 0; h < num_heads; ++h) {
      for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < total_len; ++j) {
          scores(b, h, i, j) += mask(i, j);
        }
      }
    }
  }

  // Softmax over last dimension
  scores = scores.softmax(3);

  // Attention output: scores @ V
  // scores: [batch, num_heads, seq_len, total_len]
  // V: [batch, num_kv_heads, total_len, head_dim]
  // output: [batch, num_heads, seq_len, head_dim]
  Tensor<float> attn_output = scores.matmul(v);

  // Transpose back: [batch, seq_len, num_heads, head_dim]
  attn_output = attn_output.transpose(1, 2);

  // Reshape to [batch, seq_len, hidden_size]
  attn_output = attn_output.reshape(Shape({batch, seq_len, num_heads * head_dim}));

  // Output projection
  // o_proj: [hidden_size, num_heads * head_dim]
  Tensor<float> output = attn_output.reshape(Shape({batch * seq_len, num_heads * head_dim}));
  output = output.matmul(attn.o_proj.transpose());
  output = output.reshape(Shape({batch, seq_len, hidden_size}));

  return output;
}

// ============================================================================
// Transformer block: norm -> attention -> residual -> norm -> mlp -> residual
// ============================================================================
Tensor<float> Qwen3Model::transformer_block(const Tensor<float>& hidden_states,
                                             const Qwen3Layer& layer,
                                             size_t layer_idx,
                                             size_t start_pos,
                                             KVCache* cache) const {
  // Pre-attention normalization
  Tensor<float> normed = hidden_states.rms_norm(layer.input_layernorm, config_.rms_norm_eps);

  // Attention
  Tensor<float> attn_output = attention(normed, layer.self_attn, layer_idx, start_pos, cache);

  // Residual connection
  Tensor<float> hidden = hidden_states + attn_output;

  // Pre-MLP normalization
  normed = hidden.rms_norm(layer.post_attention_layernorm, config_.rms_norm_eps);

  // MLP
  Tensor<float> mlp_output = mlp(normed, layer.mlp);

  // Residual connection
  hidden = hidden + mlp_output;

  return hidden;
}

// ============================================================================
// LM Head: project hidden states to vocabulary logits
// ============================================================================
Tensor<float> Qwen3Model::lm_head(const Tensor<float>& hidden_states) const {
  // hidden_states: [batch, seq_len, hidden_size]
  // embed_tokens: [vocab_size, hidden_size]
  // Output: [batch, seq_len, vocab_size]

  size_t batch = hidden_states.size(0);
  size_t seq_len = hidden_states.size(1);
  size_t hidden_size = hidden_states.size(2);
  size_t vocab_size = config_.vocab_size;

  // Reshape to 2D: [batch * seq_len, hidden_size]
  Tensor<float> x = hidden_states.reshape(Shape({batch * seq_len, hidden_size}));

  // Matmul with embedding weights (weight tying)
  // x @ embed_tokens.T -> [batch * seq_len, vocab_size]
  Tensor<float> logits = x.matmul(embed_tokens_.transpose());

  // Reshape back
  return logits.reshape(Shape({batch, seq_len, vocab_size}));
}

// ============================================================================
// Full forward pass
// ============================================================================
Tensor<float> Qwen3Model::forward(const std::vector<uint32_t>& tokens,
                                   size_t start_pos,
                                   KVCache* cache) const {
  // Embed tokens
  Tensor<float> hidden = embed(tokens);

  // Process through all layers
  for (size_t i = 0; i < layers_.size(); ++i) {
    hidden = transformer_block(hidden, layers_[i], i, start_pos, cache);
  }

  // Final normalization
  hidden = hidden.rms_norm(norm_, config_.rms_norm_eps);

  // Compute logits
  return lm_head(hidden);
}

// ============================================================================
// Token generation
// ============================================================================
std::vector<uint32_t> Qwen3Model::generate(const std::vector<uint32_t>& prompt,
                                            size_t max_tokens,
                                            float temperature) const {
  std::vector<uint32_t> output = prompt;
  KVCache cache;

  // Random number generator for sampling
  std::random_device rd;
  std::mt19937 gen(rd());

  // Process prompt
  Tensor<float> logits = forward(prompt, 0, &cache);

  // Get logits for last token: [1, 1, vocab_size] -> [vocab_size]
  size_t vocab_size = config_.vocab_size;
  std::vector<float> last_logits(vocab_size);
  for (size_t i = 0; i < vocab_size; ++i) {
    last_logits[i] = logits(0, prompt.size() - 1, i);
  }

  // Sample next token
  auto sample_token = [&](const std::vector<float>& logits_vec) -> uint32_t {
    if (temperature <= 0.0f) {
      // Greedy: argmax
      return static_cast<uint32_t>(
          std::max_element(logits_vec.begin(), logits_vec.end()) - logits_vec.begin());
    }

    // Apply temperature and softmax
    std::vector<float> probs(logits_vec.size());
    float max_logit = *std::max_element(logits_vec.begin(), logits_vec.end());
    float sum = 0.0f;
    for (size_t i = 0; i < logits_vec.size(); ++i) {
      probs[i] = std::exp((logits_vec[i] - max_logit) / temperature);
      sum += probs[i];
    }
    for (size_t i = 0; i < probs.size(); ++i) {
      probs[i] /= sum;
    }

    // Sample from distribution
    std::discrete_distribution<uint32_t> dist(probs.begin(), probs.end());
    return dist(gen);
  };

  uint32_t next_token = sample_token(last_logits);
  output.push_back(next_token);

  // Generate remaining tokens
  for (size_t i = 1; i < max_tokens; ++i) {
    // Forward single token with cache
    std::vector<uint32_t> single_token = {next_token};
    logits = forward(single_token, output.size() - 1, &cache);

    // Get logits
    for (size_t j = 0; j < vocab_size; ++j) {
      last_logits[j] = logits(0, 0, j);
    }

    next_token = sample_token(last_logits);
    output.push_back(next_token);

    // Stop on EOS (151645 = <|im_end|>)
    if (next_token == 151645) {
      break;
    }
  }

  return output;
}
