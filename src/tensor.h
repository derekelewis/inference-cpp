#pragma once

#include <initializer_list>
#include <vector>

class Shape {
 public:
  Shape(std::initializer_list<size_t> params);
  const std::vector<size_t>& dims() const;

 private:
  std::vector<size_t> dims_;
};

template <typename T>
class Tensor {
 public:
  Tensor(const Shape& s);
  Shape shape() const;

 private:
  Shape shape_;
  std::vector<T> data_;
};

template <typename T>
Tensor<T>::Tensor(const Shape& s) : shape_{s} {
  size_t tensor_size{1};
  for (const size_t& dim : s.dims()) {
    tensor_size *= dim;
  }
  data_ = std::vector<T>(tensor_size);
}

template <typename T>
Shape Tensor<T>::shape() const {
  return shape_;
}
