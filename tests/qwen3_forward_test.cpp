#include <gtest/gtest.h>

#include <fstream>
#include <memory>

#include "qwen3.h"

// Test path to the Qwen model
const char* QWEN_MODEL_PATH =
    "/home/dlewis/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/"
    "snapshots/cdbee75f17c01a7cc42f958dc650907174af0554";

// Shared fixture that loads model once for all tests
class Qwen3ForwardTest : public ::testing::Test {
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

std::unique_ptr<Qwen3Model> Qwen3ForwardTest::model_;
bool Qwen3ForwardTest::model_load_attempted_ = false;

// ============================================================================
// Forward Pass Tests
// ============================================================================

TEST_F(Qwen3ForwardTest, ForwardSingleToken) {
  if (!model_) {
    GTEST_SKIP() << "Model files not available";
  }

  // Token 9707 = "Hello"
  std::vector<uint32_t> tokens = {9707};
  Tensor<float> logits = model().forward(tokens);

  // Check output shape: [batch=1, seq_len=1, vocab_size]
  EXPECT_EQ(logits.ndim(), 3u);
  EXPECT_EQ(logits.size(0), 1u);
  EXPECT_EQ(logits.size(1), 1u);
  EXPECT_EQ(logits.size(2), model().config().vocab_size);

  // Logits should have reasonable values (not NaN or inf)
  float max_logit = logits.max();
  float min_logit = logits.min();
  EXPECT_FALSE(std::isnan(max_logit));
  EXPECT_FALSE(std::isinf(max_logit));
  EXPECT_FALSE(std::isnan(min_logit));
  EXPECT_FALSE(std::isinf(min_logit));

  // Max logit should be reasonable (not too extreme)
  EXPECT_LT(max_logit, 100.0f);
  EXPECT_GT(min_logit, -100.0f);
}

TEST_F(Qwen3ForwardTest, ForwardMultipleTokens) {
  if (!model_) {
    GTEST_SKIP() << "Model files not available";
  }

  // "Hello, world!" tokens (from tokenizer test)
  std::vector<uint32_t> tokens = {9707, 11, 1879, 0};
  Tensor<float> logits = model().forward(tokens);

  // Check output shape: [batch=1, seq_len=4, vocab_size]
  EXPECT_EQ(logits.ndim(), 3u);
  EXPECT_EQ(logits.size(0), 1u);
  EXPECT_EQ(logits.size(1), 4u);
  EXPECT_EQ(logits.size(2), model().config().vocab_size);

  // Logits should have reasonable values
  float max_logit = logits.max();
  float min_logit = logits.min();
  EXPECT_FALSE(std::isnan(max_logit));
  EXPECT_FALSE(std::isinf(max_logit));
  EXPECT_LT(max_logit, 100.0f);
  EXPECT_GT(min_logit, -100.0f);
}

TEST_F(Qwen3ForwardTest, ForwardWithKVCache) {
  if (!model_) {
    GTEST_SKIP() << "Model files not available";
  }

  KVCache cache;

  // First forward: process prompt
  std::vector<uint32_t> prompt = {9707};  // "Hello"
  Tensor<float> logits1 = model().forward(prompt, 0, &cache);

  // Check cache was populated
  EXPECT_EQ(cache.k_cache.size(), model().num_layers());
  EXPECT_EQ(cache.v_cache.size(), model().num_layers());

  // Second forward: single token with cache
  std::vector<uint32_t> next_token = {11};  // ","
  Tensor<float> logits2 = model().forward(next_token, 1, &cache);

  // Output shape should be for single token
  EXPECT_EQ(logits2.size(1), 1u);

  // Cache should have grown
  EXPECT_EQ(cache.k_cache[0].size(2), 2u);  // seq_len = 2 now
}

TEST_F(Qwen3ForwardTest, GenerateGreedy) {
  if (!model_) {
    GTEST_SKIP() << "Model files not available";
  }

  // Simple prompt: just the token for "1"
  std::vector<uint32_t> prompt = {16};  // "1"

  // Generate with temperature=0 (greedy)
  std::vector<uint32_t> output = model().generate(prompt, 5, 0.0f);

  // Should have prompt + generated tokens
  EXPECT_GE(output.size(), prompt.size());
  EXPECT_LE(output.size(), prompt.size() + 5);

  // First token should be the prompt
  EXPECT_EQ(output[0], prompt[0]);

  // Print generated tokens for debugging
  std::cout << "Generated tokens: ";
  for (uint32_t tok : output) {
    std::cout << tok << " ";
  }
  std::cout << std::endl;
}
