#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.h"

// ============================================================================
// TensorInfo - metadata for a single tensor in a safetensor file
// ============================================================================
struct TensorInfo {
  std::string name;
  std::string dtype;            // "BF16", "F32", "F16", etc.
  std::vector<size_t> shape;
  uint64_t data_offset_start;   // Relative to data section start
  uint64_t data_offset_end;

  size_t numel() const;
  size_t byte_size() const;
};

// ============================================================================
// BF16 conversion utilities
// ============================================================================

// Convert a single BF16 value to float32
inline float bf16_to_float(uint16_t bf16_bits) {
  // BF16 is the upper 16 bits of a float32
  uint32_t float32_bits = static_cast<uint32_t>(bf16_bits) << 16;
  float result;
  std::memcpy(&result, &float32_bits, sizeof(float));
  return result;
}

// Convert a buffer of BF16 values to float32
std::vector<float> convert_bf16_to_float32(const uint8_t* data,
                                           size_t num_elements);

// ============================================================================
// SafetensorFile - handles a single safetensor file
// ============================================================================
class SafetensorFile {
 public:
  explicit SafetensorFile(const std::string& filepath);

  // Get tensor info by name, returns nullptr if not found
  const TensorInfo* get_tensor_info(const std::string& name) const;

  // Get all tensor names in this file
  std::vector<std::string> tensor_names() const;

  // Load a specific tensor by name, converting to float32
  Tensor<float> load_tensor(const std::string& name) const;

  // Check if tensor exists
  bool has_tensor(const std::string& name) const;

  // Get filepath
  const std::string& filepath() const { return filepath_; }

 private:
  std::string filepath_;
  uint64_t header_size_;
  uint64_t data_start_offset_;  // = 8 + header_size_
  std::unordered_map<std::string, TensorInfo> tensors_;

  void parse_header();
};

// ============================================================================
// ModelLoader - handles sharded/single file models
// ============================================================================
class ModelLoader {
 public:
  // Load from a directory containing safetensor files
  explicit ModelLoader(const std::string& model_path);

  // Load tensor by name (handles sharding automatically)
  Tensor<float> load_tensor(const std::string& name) const;

  // Get all available tensor names
  std::vector<std::string> tensor_names() const;

  // Check if tensor exists
  bool has_tensor(const std::string& name) const;

 private:
  std::string model_path_;
  std::unordered_map<std::string, std::string> tensor_to_file_;  // name -> filepath
  std::unordered_map<std::string, SafetensorFile> files_;

  void load_index();
  void load_single_file(const std::string& filepath);
};

// ============================================================================
// Legacy utility functions (kept for backward compatibility)
// ============================================================================
std::vector<uint8_t> load_file(const std::string& filename);
void parse_header(const std::vector<uint8_t>& buf);
