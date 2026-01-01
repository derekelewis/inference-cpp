#include <gtest/gtest.h>

#include <fstream>
#include <memory>

#include "qwen3.h"

// Test path to the Qwen model
const char* QWEN_MODEL_PATH =
    "/home/dlewis/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/"
    "snapshots/cdbee75f17c01a7cc42f958dc650907174af0554";

// Shared fixture that loads model once for all tests
class Qwen3ModelTest : public ::testing::Test {
 protected:
  static std::unique_ptr<Qwen3Model> model_;
  static bool model_load_attempted_;

  static void SetUpTestSuite() {
    if (!model_exists()) {
      return;
    }
    model_load_attempted_ = true;
    model_.reset(new Qwen3Model(QWEN_MODEL_PATH));
  }

  static void TearDownTestSuite() {
    model_.reset();
  }

  static bool model_exists() {
    std::string index_path =
        std::string(QWEN_MODEL_PATH) + "/model.safetensors.index.json";
    std::ifstream f(index_path);
    return f.good();
  }

  const Qwen3Model& model() const {
    return *model_;
  }
};

std::unique_ptr<Qwen3Model> Qwen3ModelTest::model_;
bool Qwen3ModelTest::model_load_attempted_ = false;

// ============================================================================
// Config Tests
// ============================================================================

TEST(Qwen3ConfigTest, DefaultValues) {
  Qwen3Config config;
  EXPECT_EQ(config.vocab_size, 151936u);
  EXPECT_EQ(config.hidden_size, 2560u);
  EXPECT_EQ(config.intermediate_size, 9728u);
  EXPECT_EQ(config.num_hidden_layers, 36u);
  EXPECT_EQ(config.num_attention_heads, 32u);
  EXPECT_EQ(config.num_key_value_heads, 8u);
  EXPECT_EQ(config.head_dim, 128u);
  EXPECT_FLOAT_EQ(config.rms_norm_eps, 1e-6f);
}

// ============================================================================
// Model Loading Tests
// ============================================================================

TEST_F(Qwen3ModelTest, LoadModel) {
  if (!model_) {
    GTEST_SKIP() << "Model files not available";
  }

  // Check layer count
  EXPECT_EQ(model().num_layers(), 36u);
}

TEST_F(Qwen3ModelTest, EmbedTokensShape) {
  if (!model_) {
    GTEST_SKIP() << "Model files not available";
  }

  // embed_tokens should be [vocab_size, hidden_size]
  const Tensor<float>& embed = model().embed_tokens();
  EXPECT_EQ(embed.ndim(), 2u);
  EXPECT_EQ(embed.size(0), 151936u);  // vocab_size
  EXPECT_EQ(embed.size(1), 2560u);    // hidden_size
}

TEST_F(Qwen3ModelTest, FinalNormShape) {
  if (!model_) {
    GTEST_SKIP() << "Model files not available";
  }

  // final norm should be [hidden_size]
  const Tensor<float>& norm = model().norm();
  EXPECT_EQ(norm.ndim(), 1u);
  EXPECT_EQ(norm.size(0), 2560u);
}

TEST_F(Qwen3ModelTest, Layer0Shapes) {
  if (!model_) {
    GTEST_SKIP() << "Model files not available";
  }

  const Qwen3Layer& layer = model().layer(0);

  // Layer norms
  EXPECT_EQ(layer.input_layernorm.ndim(), 1u);
  EXPECT_EQ(layer.input_layernorm.size(0), 2560u);

  EXPECT_EQ(layer.post_attention_layernorm.ndim(), 1u);
  EXPECT_EQ(layer.post_attention_layernorm.size(0), 2560u);

  // Attention Q projection: [num_heads * head_dim, hidden_size] = [4096, 2560]
  EXPECT_EQ(layer.self_attn.q_proj.ndim(), 2u);
  EXPECT_EQ(layer.self_attn.q_proj.size(0), 4096u);
  EXPECT_EQ(layer.self_attn.q_proj.size(1), 2560u);

  // Attention K projection: [num_kv_heads * head_dim, hidden_size] = [1024,
  // 2560]
  EXPECT_EQ(layer.self_attn.k_proj.ndim(), 2u);
  EXPECT_EQ(layer.self_attn.k_proj.size(0), 1024u);
  EXPECT_EQ(layer.self_attn.k_proj.size(1), 2560u);

  // Attention V projection: [num_kv_heads * head_dim, hidden_size] = [1024,
  // 2560]
  EXPECT_EQ(layer.self_attn.v_proj.ndim(), 2u);
  EXPECT_EQ(layer.self_attn.v_proj.size(0), 1024u);
  EXPECT_EQ(layer.self_attn.v_proj.size(1), 2560u);

  // Attention O projection: [hidden_size, num_heads * head_dim] = [2560, 4096]
  EXPECT_EQ(layer.self_attn.o_proj.ndim(), 2u);
  EXPECT_EQ(layer.self_attn.o_proj.size(0), 2560u);
  EXPECT_EQ(layer.self_attn.o_proj.size(1), 4096u);

  // Q/K norms: [head_dim] = [128]
  EXPECT_EQ(layer.self_attn.q_norm.ndim(), 1u);
  EXPECT_EQ(layer.self_attn.q_norm.size(0), 128u);

  EXPECT_EQ(layer.self_attn.k_norm.ndim(), 1u);
  EXPECT_EQ(layer.self_attn.k_norm.size(0), 128u);
}

TEST_F(Qwen3ModelTest, Layer0MLPShapes) {
  if (!model_) {
    GTEST_SKIP() << "Model files not available";
  }

  const Qwen3Layer& layer = model().layer(0);

  // gate_proj: [intermediate_size, hidden_size] = [9728, 2560]
  EXPECT_EQ(layer.mlp.gate_proj.ndim(), 2u);
  EXPECT_EQ(layer.mlp.gate_proj.size(0), 9728u);
  EXPECT_EQ(layer.mlp.gate_proj.size(1), 2560u);

  // up_proj: [intermediate_size, hidden_size] = [9728, 2560]
  EXPECT_EQ(layer.mlp.up_proj.ndim(), 2u);
  EXPECT_EQ(layer.mlp.up_proj.size(0), 9728u);
  EXPECT_EQ(layer.mlp.up_proj.size(1), 2560u);

  // down_proj: [hidden_size, intermediate_size] = [2560, 9728]
  EXPECT_EQ(layer.mlp.down_proj.ndim(), 2u);
  EXPECT_EQ(layer.mlp.down_proj.size(0), 2560u);
  EXPECT_EQ(layer.mlp.down_proj.size(1), 9728u);
}

TEST_F(Qwen3ModelTest, LastLayerShapes) {
  if (!model_) {
    GTEST_SKIP() << "Model files not available";
  }

  const Qwen3Layer& layer = model().layer(35);  // Last layer (0-indexed)

  // Verify last layer has same shapes as first
  EXPECT_EQ(layer.input_layernorm.size(0), 2560u);
  EXPECT_EQ(layer.self_attn.q_proj.size(0), 4096u);
  EXPECT_EQ(layer.mlp.gate_proj.size(0), 9728u);
}

TEST_F(Qwen3ModelTest, LayerOutOfRangeThrows) {
  if (!model_) {
    GTEST_SKIP() << "Model files not available";
  }

  EXPECT_THROW(model().layer(36), std::out_of_range);
  EXPECT_THROW(model().layer(100), std::out_of_range);
}

TEST_F(Qwen3ModelTest, ConfigValues) {
  if (!model_) {
    GTEST_SKIP() << "Model files not available";
  }

  const Qwen3Config& config = model().config();

  EXPECT_EQ(config.vocab_size, 151936u);
  EXPECT_EQ(config.hidden_size, 2560u);
  EXPECT_EQ(config.num_hidden_layers, 36u);
  EXPECT_EQ(config.num_attention_heads, 32u);
  EXPECT_EQ(config.num_key_value_heads, 8u);
}

