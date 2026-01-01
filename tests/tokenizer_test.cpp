#include <gtest/gtest.h>

#include <fstream>
#include <memory>

#include "tokenizer.h"

// Path to the Qwen model tokenizer
const char* QWEN_MODEL_PATH =
    "/home/dlewis/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/"
    "snapshots/cdbee75f17c01a7cc42f958dc650907174af0554";

// Shared fixture that loads tokenizer once for all tests
class TokenizerTest : public ::testing::Test {
 protected:
  static std::unique_ptr<Tokenizer> tokenizer_;
  static bool load_attempted_;

  static void SetUpTestSuite() {
    if (!tokenizer_exists()) {
      return;
    }
    load_attempted_ = true;
    std::string path = std::string(QWEN_MODEL_PATH) + "/tokenizer.json";
    tokenizer_.reset(new Tokenizer(path));
  }

  static void TearDownTestSuite() { tokenizer_.reset(); }

  static bool tokenizer_exists() {
    std::string path = std::string(QWEN_MODEL_PATH) + "/tokenizer.json";
    std::ifstream f(path);
    return f.good();
  }

  const Tokenizer& tokenizer() const { return *tokenizer_; }
};

std::unique_ptr<Tokenizer> TokenizerTest::tokenizer_;
bool TokenizerTest::load_attempted_ = false;

// ============================================================================
// Basic Loading Tests
// ============================================================================

TEST_F(TokenizerTest, LoadTokenizer) {
  if (!tokenizer_) {
    GTEST_SKIP() << "Tokenizer files not available";
  }

  // Vocab should be around 151K tokens
  EXPECT_GT(tokenizer().vocab_size(), 150000u);
  EXPECT_LT(tokenizer().vocab_size(), 160000u);
}

TEST_F(TokenizerTest, SpecialTokens) {
  if (!tokenizer_) {
    GTEST_SKIP() << "Tokenizer files not available";
  }

  // Qwen3 special tokens
  EXPECT_EQ(tokenizer().pad_token_id(), 151643u);   // <|endoftext|>
  EXPECT_EQ(tokenizer().im_start_id(), 151644u);    // <|im_start|>
  EXPECT_EQ(tokenizer().im_end_id(), 151645u);      // <|im_end|>
  EXPECT_EQ(tokenizer().eos_token_id(), 151645u);   // Same as im_end
}

// ============================================================================
// Encoding Tests
// ============================================================================

TEST_F(TokenizerTest, EncodeSimple) {
  if (!tokenizer_) {
    GTEST_SKIP() << "Tokenizer files not available";
  }

  auto ids = tokenizer().encode("Hello");
  EXPECT_GT(ids.size(), 0u);
}

TEST_F(TokenizerTest, EncodeHelloWorld) {
  if (!tokenizer_) {
    GTEST_SKIP() << "Tokenizer files not available";
  }

  auto ids = tokenizer().encode("Hello, world!");
  EXPECT_GT(ids.size(), 0u);

  // Print tokens for debugging
  std::cout << "Tokens for 'Hello, world!': ";
  for (auto id : ids) {
    std::cout << id << " ";
  }
  std::cout << std::endl;
}

TEST_F(TokenizerTest, EncodeNumbers) {
  if (!tokenizer_) {
    GTEST_SKIP() << "Tokenizer files not available";
  }

  auto ids = tokenizer().encode("12345");
  EXPECT_GT(ids.size(), 0u);
}

TEST_F(TokenizerTest, EncodeWhitespace) {
  if (!tokenizer_) {
    GTEST_SKIP() << "Tokenizer files not available";
  }

  auto ids = tokenizer().encode("Hello world");  // Two words with space
  EXPECT_GT(ids.size(), 1u);  // Should have multiple tokens
}

// ============================================================================
// Decoding Tests
// ============================================================================

TEST_F(TokenizerTest, DecodeSimple) {
  if (!tokenizer_) {
    GTEST_SKIP() << "Tokenizer files not available";
  }

  // Encode then decode
  std::string text = "Hello";
  auto ids = tokenizer().encode(text);
  std::string decoded = tokenizer().decode(ids);
  EXPECT_EQ(decoded, text);
}

TEST_F(TokenizerTest, DecodeHelloWorld) {
  if (!tokenizer_) {
    GTEST_SKIP() << "Tokenizer files not available";
  }

  std::string text = "Hello, world!";
  auto ids = tokenizer().encode(text);
  std::string decoded = tokenizer().decode(ids);
  EXPECT_EQ(decoded, text);
}

// ============================================================================
// Round-trip Tests
// ============================================================================

TEST_F(TokenizerTest, RoundTripSimple) {
  if (!tokenizer_) {
    GTEST_SKIP() << "Tokenizer files not available";
  }

  std::vector<std::string> test_cases = {
      "Hello",
      "Hello, world!",
      "The quick brown fox jumps over the lazy dog.",
      "12345",
      "foo bar baz",
  };

  for (const auto& text : test_cases) {
    auto ids = tokenizer().encode(text);
    std::string decoded = tokenizer().decode(ids);
    EXPECT_EQ(decoded, text) << "Failed for: " << text;
  }
}

TEST_F(TokenizerTest, RoundTripPunctuation) {
  if (!tokenizer_) {
    GTEST_SKIP() << "Tokenizer files not available";
  }

  std::vector<std::string> test_cases = {
      "Hello!",
      "What?",
      "Yes...",
      "(parentheses)",
      "[brackets]",
      "{braces}",
  };

  for (const auto& text : test_cases) {
    auto ids = tokenizer().encode(text);
    std::string decoded = tokenizer().decode(ids);
    EXPECT_EQ(decoded, text) << "Failed for: " << text;
  }
}

TEST_F(TokenizerTest, RoundTripCode) {
  if (!tokenizer_) {
    GTEST_SKIP() << "Tokenizer files not available";
  }

  std::vector<std::string> test_cases = {
      "int main() { return 0; }",
      "def foo():\n    pass",
      "console.log('hello');",
  };

  for (const auto& text : test_cases) {
    auto ids = tokenizer().encode(text);
    std::string decoded = tokenizer().decode(ids);
    EXPECT_EQ(decoded, text) << "Failed for: " << text;
  }
}

// ============================================================================
// Vocabulary Access Tests
// ============================================================================

TEST_F(TokenizerTest, TokenToId) {
  if (!tokenizer_) {
    GTEST_SKIP() << "Tokenizer files not available";
  }

  // Special tokens should be accessible
  uint32_t id = tokenizer().token_to_id("<|endoftext|>");
  EXPECT_EQ(id, 151643u);
}

TEST_F(TokenizerTest, IdToToken) {
  if (!tokenizer_) {
    GTEST_SKIP() << "Tokenizer files not available";
  }

  std::string token = tokenizer().id_to_token(151643);
  EXPECT_EQ(token, "<|endoftext|>");
}

TEST_F(TokenizerTest, TokenNotFound) {
  if (!tokenizer_) {
    GTEST_SKIP() << "Tokenizer files not available";
  }

  EXPECT_THROW(tokenizer().token_to_id("nonexistent_token_xyz123"),
               std::out_of_range);
}

TEST_F(TokenizerTest, IdNotFound) {
  if (!tokenizer_) {
    GTEST_SKIP() << "Tokenizer files not available";
  }

  EXPECT_THROW(tokenizer().id_to_token(999999999), std::out_of_range);
}
