#include "tokenizer.h"

#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>
#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/unistr.h>

using json = nlohmann::json;

// Encode a unicode code point to UTF-8 string
static std::string utf8_encode(uint32_t codepoint) {
  std::string result;
  if (codepoint < 0x80) {
    result += static_cast<char>(codepoint);
  } else if (codepoint < 0x800) {
    result += static_cast<char>(0xC0 | (codepoint >> 6));
    result += static_cast<char>(0x80 | (codepoint & 0x3F));
  } else if (codepoint < 0x10000) {
    result += static_cast<char>(0xE0 | (codepoint >> 12));
    result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
    result += static_cast<char>(0x80 | (codepoint & 0x3F));
  } else {
    result += static_cast<char>(0xF0 | (codepoint >> 18));
    result += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
    result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
    result += static_cast<char>(0x80 | (codepoint & 0x3F));
  }
  return result;
}

// Extract a single UTF-8 character from string, advancing the index
static std::string extract_utf8_char(const std::string& str, size_t& idx) {
  if (idx >= str.size()) return "";

  unsigned char c = str[idx];
  size_t len = 1;

  if ((c & 0x80) == 0) {
    len = 1;
  } else if ((c & 0xE0) == 0xC0) {
    len = 2;
  } else if ((c & 0xF0) == 0xE0) {
    len = 3;
  } else if ((c & 0xF8) == 0xF0) {
    len = 4;
  }

  std::string result = str.substr(idx, len);
  idx += len;
  return result;
}

Tokenizer::Tokenizer(const std::string& tokenizer_json_path) {
  build_byte_level_tables();
  load(tokenizer_json_path);
}

void Tokenizer::build_byte_level_tables() {
  // GPT-2 style byte-to-unicode mapping
  // Printable bytes stay as their character representation
  // Non-printable bytes map to unicode 256+
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) ||
                     (b >= 174 && b <= 255);
    if (printable) {
      byte_to_unicode_[b] = std::string(1, static_cast<char>(b));
    } else {
      byte_to_unicode_[b] = utf8_encode(256 + n);
      ++n;
    }
    unicode_to_byte_[byte_to_unicode_[b]] = static_cast<uint8_t>(b);
  }
}

void Tokenizer::load(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Cannot open tokenizer file: " + path);
  }

  json data = json::parse(file);

  // Load vocabulary from model.vocab
  const auto& vocab = data["model"]["vocab"];
  for (auto it = vocab.begin(); it != vocab.end(); ++it) {
    std::string token = it.key();
    uint32_t id = it.value().get<uint32_t>();
    vocab_[token] = id;
    id_to_token_[id] = token;
  }

  // Load merges from model.merges
  // Merges can be either strings "a b" or arrays ["a", "b"]
  const auto& merges = data["model"]["merges"];
  size_t rank = 0;
  for (const auto& merge : merges) {
    std::string a, b;
    if (merge.is_string()) {
      // Format: "token1 token2"
      std::string merge_str = merge.get<std::string>();
      size_t space_pos = merge_str.find(' ');
      if (space_pos != std::string::npos) {
        a = merge_str.substr(0, space_pos);
        b = merge_str.substr(space_pos + 1);
      }
    } else if (merge.is_array() && merge.size() == 2) {
      a = merge[0].get<std::string>();
      b = merge[1].get<std::string>();
    }
    if (!a.empty() && !b.empty()) {
      merge_ranks_[{a, b}] = rank++;
    }
  }

  // Load special tokens from added_tokens
  const auto& added_tokens = data["added_tokens"];
  for (const auto& token : added_tokens) {
    std::string content = token["content"].get<std::string>();
    uint32_t id = token["id"].get<uint32_t>();
    special_tokens_[content] = id;
    vocab_[content] = id;
    id_to_token_[id] = content;
  }

  // Set common special token IDs
  auto it = special_tokens_.find("<|endoftext|>");
  pad_token_id_ = (it != special_tokens_.end()) ? it->second : 0;

  it = special_tokens_.find("<|im_start|>");
  im_start_id_ = (it != special_tokens_.end()) ? it->second : 0;

  it = special_tokens_.find("<|im_end|>");
  im_end_id_ = (it != special_tokens_.end()) ? it->second : 0;
  eos_token_id_ = im_end_id_;  // For Qwen3, <|im_end|> is the EOS token
}

std::string Tokenizer::normalize_nfc(const std::string& text) const {
  UErrorCode status = U_ZERO_ERROR;
  const icu::Normalizer2* normalizer =
      icu::Normalizer2::getNFCInstance(status);
  if (U_FAILURE(status)) {
    return text;  // Fallback to original on error
  }

  icu::UnicodeString ustr = icu::UnicodeString::fromUTF8(text);
  icu::UnicodeString normalized;
  normalizer->normalize(ustr, normalized, status);

  if (U_FAILURE(status)) {
    return text;
  }

  std::string result;
  normalized.toUTF8String(result);
  return result;
}

void Tokenizer::pre_tokenize(const std::string& text,
                             std::vector<std::string>& tokens) const {
  UErrorCode status = U_ZERO_ERROR;

  // GPT-2 style pattern for pre-tokenization
  icu::UnicodeString pattern =
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
      "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|"
      "\\p{N}|"
      " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
      "\\s*[\\r\\n]+|"
      "\\s+(?!\\S)|"
      "\\s+";

  icu::UnicodeString utext = icu::UnicodeString::fromUTF8(text);
  icu::RegexMatcher matcher(pattern, utext, 0, status);

  if (U_FAILURE(status)) {
    // Fallback: just return the whole text as one token
    tokens.push_back(text);
    return;
  }

  while (matcher.find()) {
    icu::UnicodeString match = matcher.group(status);
    if (U_FAILURE(status)) break;

    std::string token;
    match.toUTF8String(token);
    tokens.push_back(token);
  }
}

std::string Tokenizer::bytes_to_unicode(const std::string& bytes) const {
  std::string result;
  for (unsigned char byte : bytes) {
    result += byte_to_unicode_[byte];
  }
  return result;
}

std::string Tokenizer::unicode_to_bytes(const std::string& unicode_str) const {
  std::string result;
  size_t i = 0;
  while (i < unicode_str.size()) {
    std::string ch = extract_utf8_char(unicode_str, i);
    auto it = unicode_to_byte_.find(ch);
    if (it != unicode_to_byte_.end()) {
      result += static_cast<char>(it->second);
    } else {
      // Not a ByteLevel character, pass through as-is
      result += ch;
    }
  }
  return result;
}

std::vector<std::string> Tokenizer::bpe_encode(const std::string& word) const {
  // Convert word bytes to ByteLevel unicode representation
  std::vector<std::string> tokens;
  for (unsigned char byte : word) {
    tokens.push_back(byte_to_unicode_[byte]);
  }

  // Apply BPE merges until no more can be applied
  while (tokens.size() > 1) {
    // Find the pair with lowest merge rank
    size_t best_idx = std::numeric_limits<size_t>::max();
    size_t best_rank = std::numeric_limits<size_t>::max();

    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
      auto it = merge_ranks_.find({tokens[i], tokens[i + 1]});
      if (it != merge_ranks_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_idx = i;
      }
    }

    if (best_idx == std::numeric_limits<size_t>::max()) {
      break;  // No more merges possible
    }

    // Apply the merge
    tokens[best_idx] = tokens[best_idx] + tokens[best_idx + 1];
    tokens.erase(tokens.begin() + static_cast<long>(best_idx) + 1);
  }

  return tokens;
}

std::vector<uint32_t> Tokenizer::encode(const std::string& text) const {
  std::vector<uint32_t> ids;

  // Apply NFC normalization
  std::string normalized = normalize_nfc(text);

  // Check for special tokens and handle them
  // For now, we'll do a simple approach: pre-tokenize first, then encode each
  // piece

  std::vector<std::string> pre_tokens;
  pre_tokenize(normalized, pre_tokens);

  for (const auto& word : pre_tokens) {
    // Check if word is a special token
    auto special_it = special_tokens_.find(word);
    if (special_it != special_tokens_.end()) {
      ids.push_back(special_it->second);
      continue;
    }

    // Apply BPE encoding
    std::vector<std::string> bpe_tokens = bpe_encode(word);

    for (const auto& tok : bpe_tokens) {
      auto it = vocab_.find(tok);
      if (it != vocab_.end()) {
        ids.push_back(it->second);
      }
      // If not in vocab, skip (unknown token handling could be added)
    }
  }

  return ids;
}

std::string Tokenizer::decode(const std::vector<uint32_t>& ids) const {
  std::string result;

  for (uint32_t id : ids) {
    auto it = id_to_token_.find(id);
    if (it != id_to_token_.end()) {
      result += it->second;
    }
  }

  // Convert ByteLevel representation back to actual bytes
  return unicode_to_bytes(result);
}

uint32_t Tokenizer::token_to_id(const std::string& token) const {
  auto it = vocab_.find(token);
  if (it == vocab_.end()) {
    throw std::out_of_range("Token not found: " + token);
  }
  return it->second;
}

std::string Tokenizer::id_to_token(uint32_t id) const {
  auto it = id_to_token_.find(id);
  if (it == id_to_token_.end()) {
    throw std::out_of_range("Token ID not found: " + std::to_string(id));
  }
  return it->second;
}
