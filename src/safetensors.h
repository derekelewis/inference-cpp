#pragma once

#include <string>
#include <vector>

std::vector<uint8_t> load_file(const std::string& filename);

void parse_header(const std::vector<uint8_t>& buf);