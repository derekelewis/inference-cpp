#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <limits>

#include "safetensors.h"

// ============================================================================
// BF16 Conversion Tests
// ============================================================================

TEST(BF16ConversionTest, Zero) {
  // BF16 representation of 0.0f: 0x0000
  EXPECT_FLOAT_EQ(bf16_to_float(0x0000), 0.0f);
}

TEST(BF16ConversionTest, One) {
  // BF16 representation of 1.0f: 0x3F80
  EXPECT_FLOAT_EQ(bf16_to_float(0x3F80), 1.0f);
}

TEST(BF16ConversionTest, NegativeOne) {
  // BF16 representation of -1.0f: 0xBF80
  EXPECT_FLOAT_EQ(bf16_to_float(0xBF80), -1.0f);
}

TEST(BF16ConversionTest, Two) {
  // BF16 representation of 2.0f: 0x4000
  EXPECT_FLOAT_EQ(bf16_to_float(0x4000), 2.0f);
}

TEST(BF16ConversionTest, Half) {
  // BF16 representation of 0.5f: 0x3F00
  EXPECT_FLOAT_EQ(bf16_to_float(0x3F00), 0.5f);
}

TEST(BF16ConversionTest, NegativeZero) {
  // BF16 representation of -0.0f: 0x8000
  float result = bf16_to_float(0x8000);
  EXPECT_FLOAT_EQ(result, 0.0f);  // -0.0 == 0.0 in floating point
  // Check sign bit is set
  uint32_t bits;
  std::memcpy(&bits, &result, sizeof(bits));
  EXPECT_TRUE(bits & 0x80000000);
}

TEST(BF16ConversionTest, PositiveInfinity) {
  // BF16 representation of +inf: 0x7F80
  float result = bf16_to_float(0x7F80);
  EXPECT_TRUE(std::isinf(result));
  EXPECT_GT(result, 0.0f);
}

TEST(BF16ConversionTest, NegativeInfinity) {
  // BF16 representation of -inf: 0xFF80
  float result = bf16_to_float(0xFF80);
  EXPECT_TRUE(std::isinf(result));
  EXPECT_LT(result, 0.0f);
}

TEST(BF16ConversionTest, QuietNaN) {
  // BF16 representation of quiet NaN: 0x7FC0
  float result = bf16_to_float(0x7FC0);
  EXPECT_TRUE(std::isnan(result));
}

TEST(BF16ConversionTest, BulkConversion) {
  // Test bulk conversion function
  uint16_t bf16_values[] = {0x3F80, 0x4000, 0xBF80, 0x0000};  // 1.0, 2.0, -1.0, 0.0
  const uint8_t* data = reinterpret_cast<const uint8_t*>(bf16_values);

  std::vector<float> result = convert_bf16_to_float32(data, 4);

  ASSERT_EQ(result.size(), 4u);
  EXPECT_FLOAT_EQ(result[0], 1.0f);
  EXPECT_FLOAT_EQ(result[1], 2.0f);
  EXPECT_FLOAT_EQ(result[2], -1.0f);
  EXPECT_FLOAT_EQ(result[3], 0.0f);
}

// ============================================================================
// TensorInfo Tests
// ============================================================================

TEST(TensorInfoTest, Numel1D) {
  TensorInfo info;
  info.shape = {10};
  EXPECT_EQ(info.numel(), 10u);
}

TEST(TensorInfoTest, Numel2D) {
  TensorInfo info;
  info.shape = {3, 4};
  EXPECT_EQ(info.numel(), 12u);
}

TEST(TensorInfoTest, Numel3D) {
  TensorInfo info;
  info.shape = {2, 3, 4};
  EXPECT_EQ(info.numel(), 24u);
}

TEST(TensorInfoTest, NumelEmpty) {
  TensorInfo info;
  info.shape = {};
  EXPECT_EQ(info.numel(), 0u);
}

TEST(TensorInfoTest, ByteSize) {
  TensorInfo info;
  info.data_offset_start = 100;
  info.data_offset_end = 500;
  EXPECT_EQ(info.byte_size(), 400u);
}

// ============================================================================
// SafetensorFile Tests (with real model files)
// ============================================================================

// Test path to the Qwen model
const char* QWEN_MODEL_PATH = "/home/dlewis/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554";

class SafetensorFileTest : public ::testing::Test {
 protected:
  static bool model_exists() {
    std::string shard_path = std::string(QWEN_MODEL_PATH) + "/model-00001-of-00003.safetensors";
    std::ifstream f(shard_path);
    return f.good();
  }
};

TEST_F(SafetensorFileTest, ParseHeader) {
  if (!model_exists()) {
    GTEST_SKIP() << "Model files not available";
  }

  std::string shard_path = std::string(QWEN_MODEL_PATH) + "/model-00001-of-00003.safetensors";
  SafetensorFile sf(shard_path);

  // Should have tensors
  std::vector<std::string> names = sf.tensor_names();
  EXPECT_GT(names.size(), 0u);

  // Check that embed_tokens exists
  EXPECT_TRUE(sf.has_tensor("model.embed_tokens.weight"));
}

TEST_F(SafetensorFileTest, GetTensorInfo) {
  if (!model_exists()) {
    GTEST_SKIP() << "Model files not available";
  }

  std::string shard_path = std::string(QWEN_MODEL_PATH) + "/model-00001-of-00003.safetensors";
  SafetensorFile sf(shard_path);

  const TensorInfo* info = sf.get_tensor_info("model.embed_tokens.weight");
  ASSERT_NE(info, nullptr);

  // Check shape: [vocab_size=151936, hidden_size=2560]
  ASSERT_EQ(info->shape.size(), 2u);
  EXPECT_EQ(info->shape[0], 151936u);
  EXPECT_EQ(info->shape[1], 2560u);

  // Check dtype
  EXPECT_EQ(info->dtype, "BF16");
}

TEST_F(SafetensorFileTest, LoadTensor) {
  if (!model_exists()) {
    GTEST_SKIP() << "Model files not available";
  }

  std::string shard_path = std::string(QWEN_MODEL_PATH) + "/model-00001-of-00003.safetensors";
  SafetensorFile sf(shard_path);

  // Load a small tensor (layer norm weight, shape [2560])
  Tensor<float> t = sf.load_tensor("model.layers.0.input_layernorm.weight");

  EXPECT_EQ(t.ndim(), 1u);
  EXPECT_EQ(t.size(0), 2560u);
  EXPECT_EQ(t.numel(), 2560u);

  // Values should be finite
  for (size_t i = 0; i < t.numel(); ++i) {
    EXPECT_TRUE(std::isfinite(t(i))) << "Value at index " << i << " is not finite";
  }
}

TEST_F(SafetensorFileTest, LoadTensorNotFound) {
  if (!model_exists()) {
    GTEST_SKIP() << "Model files not available";
  }

  std::string shard_path = std::string(QWEN_MODEL_PATH) + "/model-00001-of-00003.safetensors";
  SafetensorFile sf(shard_path);

  EXPECT_THROW(sf.load_tensor("nonexistent.tensor"), std::runtime_error);
}

// ============================================================================
// ModelLoader Tests (with real model files)
// ============================================================================

class ModelLoaderTest : public ::testing::Test {
 protected:
  static bool model_exists() {
    std::string index_path = std::string(QWEN_MODEL_PATH) + "/model.safetensors.index.json";
    std::ifstream f(index_path);
    return f.good();
  }
};

TEST_F(ModelLoaderTest, LoadShardedModel) {
  if (!model_exists()) {
    GTEST_SKIP() << "Model files not available";
  }

  ModelLoader loader(QWEN_MODEL_PATH);

  // Check tensor count
  std::vector<std::string> names = loader.tensor_names();
  EXPECT_GT(names.size(), 0u);

  // Check that key tensors exist
  EXPECT_TRUE(loader.has_tensor("model.embed_tokens.weight"));
  EXPECT_TRUE(loader.has_tensor("model.norm.weight"));
  EXPECT_TRUE(loader.has_tensor("model.layers.0.input_layernorm.weight"));
  EXPECT_TRUE(loader.has_tensor("model.layers.35.input_layernorm.weight"));  // Last layer
}

TEST_F(ModelLoaderTest, LoadTensorFromFirstShard) {
  if (!model_exists()) {
    GTEST_SKIP() << "Model files not available";
  }

  ModelLoader loader(QWEN_MODEL_PATH);

  // Load a tensor from the first shard
  Tensor<float> t = loader.load_tensor("model.layers.0.input_layernorm.weight");

  EXPECT_EQ(t.ndim(), 1u);
  EXPECT_EQ(t.size(0), 2560u);
}

TEST_F(ModelLoaderTest, LoadTensorFromLastShard) {
  if (!model_exists()) {
    GTEST_SKIP() << "Model files not available";
  }

  ModelLoader loader(QWEN_MODEL_PATH);

  // Load a tensor from the last shard (model.norm.weight is in shard 3)
  Tensor<float> t = loader.load_tensor("model.norm.weight");

  EXPECT_EQ(t.ndim(), 1u);
  EXPECT_EQ(t.size(0), 2560u);  // hidden_size
}

TEST_F(ModelLoaderTest, LoadEmbeddingTensorShape) {
  if (!model_exists()) {
    GTEST_SKIP() << "Model files not available";
  }

  ModelLoader loader(QWEN_MODEL_PATH);

  Tensor<float> t = loader.load_tensor("model.embed_tokens.weight");

  // Shape should be [vocab_size, hidden_size] = [151936, 2560]
  EXPECT_EQ(t.ndim(), 2u);
  EXPECT_EQ(t.size(0), 151936u);
  EXPECT_EQ(t.size(1), 2560u);
}

TEST_F(ModelLoaderTest, LoadAttentionProjectionShape) {
  if (!model_exists()) {
    GTEST_SKIP() << "Model files not available";
  }

  ModelLoader loader(QWEN_MODEL_PATH);

  // q_proj has shape [num_heads * head_dim, hidden_size] = [32*128, 2560] = [4096, 2560]
  Tensor<float> q_proj = loader.load_tensor("model.layers.0.self_attn.q_proj.weight");
  EXPECT_EQ(q_proj.ndim(), 2u);
  EXPECT_EQ(q_proj.size(0), 4096u);
  EXPECT_EQ(q_proj.size(1), 2560u);

  // k_proj has shape [num_kv_heads * head_dim, hidden_size] = [8*128, 2560] = [1024, 2560]
  Tensor<float> k_proj = loader.load_tensor("model.layers.0.self_attn.k_proj.weight");
  EXPECT_EQ(k_proj.ndim(), 2u);
  EXPECT_EQ(k_proj.size(0), 1024u);
  EXPECT_EQ(k_proj.size(1), 2560u);
}

TEST_F(ModelLoaderTest, LoadMLPProjectionShape) {
  if (!model_exists()) {
    GTEST_SKIP() << "Model files not available";
  }

  ModelLoader loader(QWEN_MODEL_PATH);

  // gate_proj has shape [intermediate_size, hidden_size] = [9728, 2560]
  Tensor<float> gate_proj = loader.load_tensor("model.layers.0.mlp.gate_proj.weight");
  EXPECT_EQ(gate_proj.ndim(), 2u);
  EXPECT_EQ(gate_proj.size(0), 9728u);
  EXPECT_EQ(gate_proj.size(1), 2560u);

  // down_proj has shape [hidden_size, intermediate_size] = [2560, 9728]
  Tensor<float> down_proj = loader.load_tensor("model.layers.0.mlp.down_proj.weight");
  EXPECT_EQ(down_proj.ndim(), 2u);
  EXPECT_EQ(down_proj.size(0), 2560u);
  EXPECT_EQ(down_proj.size(1), 9728u);
}

TEST_F(ModelLoaderTest, TensorNotFound) {
  if (!model_exists()) {
    GTEST_SKIP() << "Model files not available";
  }

  ModelLoader loader(QWEN_MODEL_PATH);

  EXPECT_THROW(loader.load_tensor("nonexistent.tensor"), std::runtime_error);
}
