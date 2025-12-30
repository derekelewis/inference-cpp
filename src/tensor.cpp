#include "tensor.h"

#include <initializer_list>
#include <stdexcept>

Shape::Shape(std::initializer_list<size_t> params) {
  if (params.size() < 1)
    throw std::runtime_error("invalid shape dimensions specified");
  dims_ = std::vector<size_t>{params};
}

const std::vector<size_t>& Shape::dims() const { return dims_; }
