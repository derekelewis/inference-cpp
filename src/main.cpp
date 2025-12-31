#include <stdexcept>
#include <string>

#include "safetensors.h"
#include "tensor.h"

int main(int argc, char* argv[]) {
  if (argc < 2) throw std::runtime_error("no tensor file specified");
  const std::string tensor_file{argv[1]};
  std::vector<uint8_t> buf = load_file(tensor_file);
  parse_header(buf);
  Tensor<float> tensor{Shape{2, 3}};
  return 0;
}