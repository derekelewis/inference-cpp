#include "qwen3.h"

#include <stdexcept>
#include <string>

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
