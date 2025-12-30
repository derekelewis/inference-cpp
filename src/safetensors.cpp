#include "safetensors.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

using json = nlohmann::json;

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
