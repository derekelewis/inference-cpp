#include "safetensors.h"

#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <vector>

using json = nlohmann::json;

// ============================================================================
// TensorInfo implementation
// ============================================================================

size_t TensorInfo::numel() const {
  if (shape.empty()) return 0;
  size_t n = 1;
  for (size_t d : shape) n *= d;
  return n;
}

size_t TensorInfo::byte_size() const {
  return static_cast<size_t>(data_offset_end - data_offset_start);
}

// ============================================================================
// BF16 conversion utilities
// ============================================================================

std::vector<float> convert_bf16_to_float32(const uint8_t* data,
                                           size_t num_elements) {
  std::vector<float> result(num_elements);
  const uint16_t* bf16_data = reinterpret_cast<const uint16_t*>(data);

  for (size_t i = 0; i < num_elements; ++i) {
    result[i] = bf16_to_float(bf16_data[i]);
  }

  return result;
}

// ============================================================================
// SafetensorFile implementation
// ============================================================================

SafetensorFile::SafetensorFile(const std::string& filepath)
    : filepath_(filepath), header_size_(0), data_start_offset_(0) {
  parse_header();
}

void SafetensorFile::parse_header() {
  std::ifstream file(filepath_, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open: " + filepath_);
  }

  // Read header size (8 bytes, little-endian)
  uint8_t size_buf[8];
  if (!file.read(reinterpret_cast<char*>(size_buf), 8)) {
    throw std::runtime_error("Failed to read header size: " + filepath_);
  }

  header_size_ = 0;
  for (int i = 0; i < 8; ++i) {
    header_size_ |= static_cast<uint64_t>(size_buf[i]) << (i * 8);
  }
  data_start_offset_ = 8 + header_size_;

  // Read header JSON
  std::string header_json(static_cast<size_t>(header_size_), '\0');
  if (!file.read(&header_json[0],
                 static_cast<std::streamsize>(header_size_))) {
    throw std::runtime_error("Failed to read header JSON: " + filepath_);
  }

  // Parse JSON
  json header_data = json::parse(header_json);

  // Extract tensor info
  for (auto it = header_data.begin(); it != header_data.end(); ++it) {
    const std::string& key = it.key();
    if (key == "__metadata__") continue;  // Skip metadata

    const json& value = it.value();

    TensorInfo info;
    info.name = key;
    info.dtype = value["dtype"].get<std::string>();

    // Parse shape
    const json& shape_arr = value["shape"];
    info.shape.reserve(shape_arr.size());
    for (size_t i = 0; i < shape_arr.size(); ++i) {
      info.shape.push_back(shape_arr[i].get<size_t>());
    }

    // Parse data offsets
    const json& offsets = value["data_offsets"];
    info.data_offset_start = offsets[0].get<uint64_t>();
    info.data_offset_end = offsets[1].get<uint64_t>();

    tensors_[key] = info;
  }
}

const TensorInfo* SafetensorFile::get_tensor_info(
    const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    return nullptr;
  }
  return &it->second;
}

std::vector<std::string> SafetensorFile::tensor_names() const {
  std::vector<std::string> names;
  names.reserve(tensors_.size());
  for (auto it = tensors_.begin(); it != tensors_.end(); ++it) {
    names.push_back(it->first);
  }
  return names;
}

bool SafetensorFile::has_tensor(const std::string& name) const {
  return tensors_.find(name) != tensors_.end();
}

Tensor<float> SafetensorFile::load_tensor(const std::string& name) const {
  const TensorInfo* info = get_tensor_info(name);
  if (!info) {
    throw std::runtime_error("Tensor not found: " + name);
  }

  std::ifstream file(filepath_, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open: " + filepath_);
  }

  // Seek to tensor data position
  uint64_t absolute_offset = data_start_offset_ + info->data_offset_start;
  file.seekg(static_cast<std::streamoff>(absolute_offset));

  size_t byte_size = info->byte_size();
  std::vector<uint8_t> raw_data(byte_size);

  if (!file.read(reinterpret_cast<char*>(raw_data.data()),
                 static_cast<std::streamsize>(byte_size))) {
    throw std::runtime_error("Failed to read tensor data: " + name);
  }

  // Convert based on dtype
  std::vector<float> float_data;
  if (info->dtype == "BF16") {
    float_data = convert_bf16_to_float32(raw_data.data(), info->numel());
  } else if (info->dtype == "F32") {
    float_data.resize(info->numel());
    std::memcpy(float_data.data(), raw_data.data(), byte_size);
  } else if (info->dtype == "F16") {
    throw std::runtime_error("F16 dtype not yet supported: " + name);
  } else {
    throw std::runtime_error("Unsupported dtype: " + info->dtype);
  }

  return Tensor<float>(Shape(info->shape), float_data);
}

// ============================================================================
// ModelLoader implementation
// ============================================================================

ModelLoader::ModelLoader(const std::string& model_path)
    : model_path_(model_path) {
  load_index();
}

void ModelLoader::load_single_file(const std::string& filepath) {
  // Use emplace with piecewise_construct for C++11 compatibility
  auto result = files_.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(filepath),
      std::forward_as_tuple(filepath));

  if (result.second) {
    // New file was inserted, add its tensors to the mapping
    const SafetensorFile& sf = result.first->second;
    std::vector<std::string> names = sf.tensor_names();
    for (size_t i = 0; i < names.size(); ++i) {
      tensor_to_file_[names[i]] = filepath;
    }
  }
}

void ModelLoader::load_index() {
  // Try to load model.safetensors.index.json for sharded models
  std::string index_path = model_path_ + "/model.safetensors.index.json";
  std::ifstream index_file(index_path);

  if (index_file) {
    // Sharded model with index file
    json index = json::parse(index_file);
    json weight_map = index["weight_map"];

    std::set<std::string> shard_files;
    for (auto it = weight_map.begin(); it != weight_map.end(); ++it) {
      const std::string& tensor_name = it.key();
      std::string filename = it.value().get<std::string>();
      std::string full_path = model_path_ + "/" + filename;
      tensor_to_file_[tensor_name] = full_path;
      shard_files.insert(full_path);
    }

    // Load all shard files (headers only, not data)
    for (std::set<std::string>::const_iterator it = shard_files.begin();
         it != shard_files.end(); ++it) {
      files_.emplace(
          std::piecewise_construct,
          std::forward_as_tuple(*it),
          std::forward_as_tuple(*it));
    }
  } else {
    // Try single file model
    std::string single_file = model_path_ + "/model.safetensors";
    std::ifstream check_file(single_file);
    if (check_file) {
      check_file.close();
      load_single_file(single_file);
    } else {
      // model_path might be a direct path to a safetensor file
      std::ifstream direct_file(model_path_);
      if (direct_file) {
        direct_file.close();
        load_single_file(model_path_);
      } else {
        throw std::runtime_error(
            "No safetensor files found at: " + model_path_);
      }
    }
  }
}

Tensor<float> ModelLoader::load_tensor(const std::string& name) const {
  auto it = tensor_to_file_.find(name);
  if (it == tensor_to_file_.end()) {
    throw std::runtime_error("Tensor not found in model: " + name);
  }

  auto file_it = files_.find(it->second);
  if (file_it == files_.end()) {
    throw std::runtime_error("Shard file not loaded: " + it->second);
  }

  return file_it->second.load_tensor(name);
}

std::vector<std::string> ModelLoader::tensor_names() const {
  std::vector<std::string> names;
  names.reserve(tensor_to_file_.size());
  for (auto it = tensor_to_file_.begin(); it != tensor_to_file_.end(); ++it) {
    names.push_back(it->first);
  }
  return names;
}

bool ModelLoader::has_tensor(const std::string& name) const {
  return tensor_to_file_.find(name) != tensor_to_file_.end();
}

// ============================================================================
// Legacy utility functions (kept for backward compatibility)
// ============================================================================

std::vector<uint8_t> load_file(const std::string& filename) {
  std::ifstream input{filename, std::ios::binary};
  if (!input) throw std::runtime_error("failed to open: " + filename);

  input.seekg(0, std::ios::end);
  std::streamsize size = input.tellg();
  input.seekg(0, std::ios::beg);

  if (size < 0) throw std::runtime_error("tellg failed: " + filename);

  std::vector<uint8_t> buf(static_cast<size_t>(size));
  if (!input.read(reinterpret_cast<char*>(buf.data()), size)) {
    throw std::runtime_error("failed to read: " + filename);
  }

  return buf;
}

void parse_header(const std::vector<uint8_t>& buf) {
  if (buf.size() < 8) throw std::runtime_error("header truncated");

  uint64_t headerSize{};
  for (size_t i = 0; i < 8; ++i) {
    const uint64_t b = buf[i];
    headerSize |= b << (i * 8);
  }

  if (buf.size() < 8 + headerSize) {
    throw std::runtime_error("file too small for header");
  }

  std::string header{reinterpret_cast<const char*>(&buf[8]),
                     static_cast<size_t>(headerSize)};
  json header_data = json::parse(header);
}
