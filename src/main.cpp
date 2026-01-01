#include <stdexcept>
#include <string>

#include "safetensors.h"
#include "tensor.h"

int main(int argc, char* argv[]) {
  if (argc < 2) throw std::runtime_error("no tensor file specified");
  ModelLoader model{argv[1]};
  return 0;
}