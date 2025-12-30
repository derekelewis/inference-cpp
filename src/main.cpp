#include <string>

#include "safetensors.h"
#include "tensor.h"

int main() {
  const std::string tensor_file{
      "/Users/dlewis/.cache/huggingface/hub/"
      "models--Qwen--Qwen3-4B-Instruct-2507/snapshots/"
      "cdbee75f17c01a7cc42f958dc650907174af0554/"
      "model-00001-of-00003.safetensors"};
  std::vector<uint8_t> buf = load_file(tensor_file);
  parse_header(buf);
  Tensor<float> tensor{Shape{2, 3}};
  return 0;
}