#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Hash function for std::pair to use in unordered_map
struct PairHash {
  template <typename T1, typename T2>
  size_t operator()(const std::pair<T1, T2>& p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);
    return h1 ^ (h2 << 1);
  }
};

class Tokenizer {
 public:
  // Load tokenizer from tokenizer.json path
  explicit Tokenizer(const std::string& tokenizer_json_path);

  // Encode text to token IDs
  std::vector<uint32_t> encode(const std::string& text) const;

  // Decode token IDs to text
  std::string decode(const std::vector<uint32_t>& ids) const;

  // Vocabulary access
  size_t vocab_size() const { return vocab_.size(); }
  uint32_t token_to_id(const std::string& token) const;
  std::string id_to_token(uint32_t id) const;

  // Special tokens
  uint32_t eos_token_id() const { return eos_token_id_; }
  uint32_t pad_token_id() const { return pad_token_id_; }
  uint32_t im_start_id() const { return im_start_id_; }
  uint32_t im_end_id() const { return im_end_id_; }

 private:
  // Vocabulary
  std::unordered_map<std::string, uint32_t> vocab_;
  std::unordered_map<uint32_t, std::string> id_to_token_;

  // BPE merges with rank lookup
  std::unordered_map<std::pair<std::string, std::string>, size_t, PairHash>
      merge_ranks_;

  // Special tokens
  std::unordered_map<std::string, uint32_t> special_tokens_;
  uint32_t eos_token_id_;
  uint32_t pad_token_id_;
  uint32_t im_start_id_;
  uint32_t im_end_id_;

  // ByteLevel encoding tables (GPT-2 style)
  std::array<std::string, 256> byte_to_unicode_;
  std::unordered_map<std::string, uint8_t> unicode_to_byte_;

  // Pre-tokenize text into words using regex
  void pre_tokenize(const std::string& text,
                    std::vector<std::string>& tokens) const;

  // Apply BPE to a single word (after ByteLevel encoding)
  std::vector<std::string> bpe_encode(const std::string& word) const;

  // Load from tokenizer.json
  void load(const std::string& path);

  // Build GPT-2 style byte-to-unicode tables
  void build_byte_level_tables();

  // NFC normalize text
  std::string normalize_nfc(const std::string& text) const;

  // Convert bytes to ByteLevel representation
  std::string bytes_to_unicode(const std::string& bytes) const;

  // Convert ByteLevel representation back to bytes
  std::string unicode_to_bytes(const std::string& unicode_str) const;
};
