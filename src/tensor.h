#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

// ============================================================================
// Shape class - represents tensor dimensions
// ============================================================================
class Shape {
 public:
  Shape() = default;
  Shape(std::initializer_list<size_t> params);
  Shape(const std::vector<size_t>& dims);

  const std::vector<size_t>& dims() const;
  size_t operator[](size_t idx) const;
  size_t ndim() const;
  size_t numel() const;

  bool operator==(const Shape& other) const;
  bool operator!=(const Shape& other) const;

  // Broadcasting support
  static bool can_broadcast(const Shape& a, const Shape& b);
  static Shape broadcast_shape(const Shape& a, const Shape& b);

 private:
  std::vector<size_t> dims_;
};

// ============================================================================
// Tensor class - multi-dimensional array with operations for neural networks
// ============================================================================
template <typename T>
class Tensor {
 public:
  // Constructors
  Tensor();
  explicit Tensor(const Shape& s);
  Tensor(const Shape& s, const T& fill_value);
  Tensor(const Shape& s, const std::vector<T>& data);
  Tensor(const Tensor& other);
  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(const Tensor& other);
  Tensor& operator=(Tensor&& other) noexcept;

  // Basic accessors
  const Shape& shape() const;
  size_t ndim() const;
  size_t numel() const;
  size_t size(size_t dim) const;

  // Data access
  T* data();
  const T* data() const;
  T& operator()(size_t i);
  const T& operator()(size_t i) const;
  T& operator()(size_t i, size_t j);
  const T& operator()(size_t i, size_t j) const;
  T& operator()(size_t i, size_t j, size_t k);
  const T& operator()(size_t i, size_t j, size_t k) const;
  T& operator()(size_t i, size_t j, size_t k, size_t l);
  const T& operator()(size_t i, size_t j, size_t k, size_t l) const;
  T& at(const std::vector<size_t>& indices);
  const T& at(const std::vector<size_t>& indices) const;

  // Shape manipulation
  Tensor<T> reshape(const Shape& new_shape) const;
  Tensor<T> transpose() const;                          // 2D transpose
  Tensor<T> transpose(size_t dim0, size_t dim1) const;  // swap two dimensions
  Tensor<T> permute(const std::vector<size_t>& dims) const;
  Tensor<T> squeeze(size_t dim) const;
  Tensor<T> unsqueeze(size_t dim) const;
  Tensor<T> slice(size_t dim, size_t start, size_t end) const;
  Tensor<T> contiguous() const;

  // Element-wise operations (returns new tensor)
  Tensor<T> operator+(const Tensor<T>& other) const;
  Tensor<T> operator-(const Tensor<T>& other) const;
  Tensor<T> operator*(const Tensor<T>& other) const;
  Tensor<T> operator/(const Tensor<T>& other) const;
  Tensor<T> operator+(const T& scalar) const;
  Tensor<T> operator-(const T& scalar) const;
  Tensor<T> operator*(const T& scalar) const;
  Tensor<T> operator/(const T& scalar) const;
  Tensor<T> neg() const;

  // In-place element-wise operations
  Tensor<T>& operator+=(const Tensor<T>& other);
  Tensor<T>& operator-=(const Tensor<T>& other);
  Tensor<T>& operator*=(const Tensor<T>& other);
  Tensor<T>& operator/=(const Tensor<T>& other);
  Tensor<T>& operator+=(const T& scalar);
  Tensor<T>& operator-=(const T& scalar);
  Tensor<T>& operator*=(const T& scalar);
  Tensor<T>& operator/=(const T& scalar);

  // Activation functions
  Tensor<T> relu() const;
  Tensor<T> gelu() const;
  Tensor<T> silu() const;  // Also known as swish
  Tensor<T> sigmoid() const;
  Tensor<T> tanh_() const;  // tanh_ to avoid conflict with std::tanh

  // Reduction operations
  T sum() const;
  T mean() const;
  T max() const;
  T min() const;
  Tensor<T> sum(size_t dim, bool keepdim = false) const;
  Tensor<T> mean(size_t dim, bool keepdim = false) const;
  Tensor<T> max(size_t dim, bool keepdim = false) const;
  Tensor<T> min(size_t dim, bool keepdim = false) const;
  Tensor<size_t> argmax(size_t dim) const;
  Tensor<size_t> argmin(size_t dim) const;

  // Matrix operations
  Tensor<T> matmul(const Tensor<T>& other) const;

  // Normalization operations
  Tensor<T> softmax(size_t dim) const;
  Tensor<T> layer_norm(const Tensor<T>& weight, const Tensor<T>& bias,
                       T eps = 1e-5) const;
  Tensor<T> rms_norm(const Tensor<T>& weight, T eps = 1e-6) const;

  // Utility functions
  void fill(const T& value);
  Tensor<T> clone() const;

  // Static factory functions
  static Tensor<T> zeros(const Shape& s);
  static Tensor<T> ones(const Shape& s);
  static Tensor<T> full(const Shape& s, const T& value);

  // Apply function element-wise
  Tensor<T> apply(std::function<T(T)> func) const;

 private:
  Shape shape_;
  std::vector<T> data_;

  // Helper functions
  size_t compute_flat_index(const std::vector<size_t>& indices) const;
  std::vector<size_t> compute_strides() const;

  // Broadcasting helper
  template <typename Op>
  Tensor<T> broadcast_op(const Tensor<T>& other, Op op) const;
};

// ============================================================================
// Shape implementation
// ============================================================================
inline Shape::Shape(std::initializer_list<size_t> params) {
  if (params.size() < 1)
    throw std::runtime_error("invalid shape dimensions specified");
  dims_ = std::vector<size_t>{params};
}

inline Shape::Shape(const std::vector<size_t>& dims) : dims_(dims) {
  if (dims_.empty())
    throw std::runtime_error("invalid shape dimensions specified");
}

inline const std::vector<size_t>& Shape::dims() const { return dims_; }

inline size_t Shape::operator[](size_t idx) const {
  if (idx >= dims_.size()) throw std::out_of_range("Shape index out of range");
  return dims_[idx];
}

inline size_t Shape::ndim() const { return dims_.size(); }

inline size_t Shape::numel() const {
  if (dims_.empty()) return 0;
  size_t total = 1;
  for (size_t d : dims_) total *= d;
  return total;
}

inline bool Shape::operator==(const Shape& other) const {
  return dims_ == other.dims_;
}

inline bool Shape::operator!=(const Shape& other) const {
  return !(*this == other);
}

inline bool Shape::can_broadcast(const Shape& a, const Shape& b) {
  const auto& a_dims = a.dims();
  const auto& b_dims = b.dims();
  size_t max_ndim = std::max(a_dims.size(), b_dims.size());

  for (size_t i = 0; i < max_ndim; ++i) {
    size_t a_dim = (i < a_dims.size()) ? a_dims[a_dims.size() - 1 - i] : 1;
    size_t b_dim = (i < b_dims.size()) ? b_dims[b_dims.size() - 1 - i] : 1;
    if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
      return false;
    }
  }
  return true;
}

inline Shape Shape::broadcast_shape(const Shape& a, const Shape& b) {
  if (!can_broadcast(a, b)) {
    throw std::runtime_error("Shapes cannot be broadcast together");
  }

  const auto& a_dims = a.dims();
  const auto& b_dims = b.dims();
  size_t max_ndim = std::max(a_dims.size(), b_dims.size());
  std::vector<size_t> result_dims(max_ndim);

  for (size_t i = 0; i < max_ndim; ++i) {
    size_t a_dim = (i < a_dims.size()) ? a_dims[a_dims.size() - 1 - i] : 1;
    size_t b_dim = (i < b_dims.size()) ? b_dims[b_dims.size() - 1 - i] : 1;
    result_dims[max_ndim - 1 - i] = std::max(a_dim, b_dim);
  }

  return Shape(result_dims);
}

// ============================================================================
// Tensor implementation
// ============================================================================

// Constructors
template <typename T>
Tensor<T>::Tensor() : shape_({1}), data_(1) {}

template <typename T>
Tensor<T>::Tensor(const Shape& s) : shape_(s), data_(s.numel()) {}

template <typename T>
Tensor<T>::Tensor(const Shape& s, const T& fill_value)
    : shape_(s), data_(s.numel(), fill_value) {}

template <typename T>
Tensor<T>::Tensor(const Shape& s, const std::vector<T>& data)
    : shape_(s), data_(data) {
  if (data_.size() != s.numel()) {
    throw std::runtime_error("Data size does not match shape size");
  }
}

template <typename T>
Tensor<T>::Tensor(const Tensor& other)
    : shape_(other.shape_), data_(other.data_) {}

template <typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), data_(std::move(other.data_)) {}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& other) {
  if (this != &other) {
    shape_ = other.shape_;
    data_ = other.data_;
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    shape_ = std::move(other.shape_);
    data_ = std::move(other.data_);
  }
  return *this;
}

// Basic accessors
template <typename T>
const Shape& Tensor<T>::shape() const {
  return shape_;
}

template <typename T>
size_t Tensor<T>::ndim() const {
  return shape_.ndim();
}

template <typename T>
size_t Tensor<T>::numel() const {
  return data_.size();
}

template <typename T>
size_t Tensor<T>::size(size_t dim) const {
  return shape_[dim];
}

// Data access
template <typename T>
T* Tensor<T>::data() {
  return data_.data();
}

template <typename T>
const T* Tensor<T>::data() const {
  return data_.data();
}

template <typename T>
T& Tensor<T>::operator()(size_t i) {
  return data_[i];
}

template <typename T>
const T& Tensor<T>::operator()(size_t i) const {
  return data_[i];
}

template <typename T>
T& Tensor<T>::operator()(size_t i, size_t j) {
  size_t cols = shape_[1];
  return data_[i * cols + j];
}

template <typename T>
const T& Tensor<T>::operator()(size_t i, size_t j) const {
  size_t cols = shape_[1];
  return data_[i * cols + j];
}

template <typename T>
T& Tensor<T>::operator()(size_t i, size_t j, size_t k) {
  const auto& dims = shape_.dims();
  return data_[(i * dims[1] + j) * dims[2] + k];
}

template <typename T>
const T& Tensor<T>::operator()(size_t i, size_t j, size_t k) const {
  const auto& dims = shape_.dims();
  return data_[(i * dims[1] + j) * dims[2] + k];
}

template <typename T>
T& Tensor<T>::operator()(size_t i, size_t j, size_t k, size_t l) {
  const auto& dims = shape_.dims();
  return data_[((i * dims[1] + j) * dims[2] + k) * dims[3] + l];
}

template <typename T>
const T& Tensor<T>::operator()(size_t i, size_t j, size_t k, size_t l) const {
  const auto& dims = shape_.dims();
  return data_[((i * dims[1] + j) * dims[2] + k) * dims[3] + l];
}

template <typename T>
size_t Tensor<T>::compute_flat_index(const std::vector<size_t>& indices) const {
  const auto& dims = shape_.dims();
  if (indices.size() != dims.size()) {
    throw std::runtime_error("Index dimensions don't match tensor dimensions");
  }
  size_t flat_idx = 0;
  size_t stride = 1;
  for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
    flat_idx += indices[i] * stride;
    stride *= dims[i];
  }
  return flat_idx;
}

template <typename T>
std::vector<size_t> Tensor<T>::compute_strides() const {
  const auto& dims = shape_.dims();
  std::vector<size_t> strides(dims.size());
  size_t stride = 1;
  for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= dims[i];
  }
  return strides;
}

template <typename T>
T& Tensor<T>::at(const std::vector<size_t>& indices) {
  return data_[compute_flat_index(indices)];
}

template <typename T>
const T& Tensor<T>::at(const std::vector<size_t>& indices) const {
  return data_[compute_flat_index(indices)];
}

// Shape manipulation
template <typename T>
Tensor<T> Tensor<T>::reshape(const Shape& new_shape) const {
  if (new_shape.numel() != numel()) {
    throw std::runtime_error(
        "Cannot reshape: total number of elements must remain the same");
  }
  Tensor<T> result(new_shape);
  std::copy(data_.begin(), data_.end(), result.data_.begin());
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::transpose() const {
  if (ndim() != 2) {
    throw std::runtime_error("transpose() without args requires 2D tensor");
  }
  size_t rows = shape_[0];
  size_t cols = shape_[1];
  Tensor<T> result(Shape({cols, rows}));
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      result(j, i) = (*this)(i, j);
    }
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::transpose(size_t dim0, size_t dim1) const {
  std::vector<size_t> perm(ndim());
  for (size_t i = 0; i < ndim(); ++i) perm[i] = i;
  std::swap(perm[dim0], perm[dim1]);
  return permute(perm);
}

template <typename T>
Tensor<T> Tensor<T>::permute(const std::vector<size_t>& dims_order) const {
  if (dims_order.size() != ndim()) {
    throw std::runtime_error("Permute dimensions must match tensor dimensions");
  }

  const auto& old_dims = shape_.dims();
  std::vector<size_t> new_dims(ndim());
  for (size_t i = 0; i < ndim(); ++i) {
    new_dims[i] = old_dims[dims_order[i]];
  }

  Tensor<T> result{Shape(new_dims)};
  std::vector<size_t> old_strides = compute_strides();

  std::vector<size_t> new_strides(ndim());
  size_t stride = 1;
  for (int i = static_cast<int>(ndim()) - 1; i >= 0; --i) {
    new_strides[i] = stride;
    stride *= new_dims[i];
  }

  // Iterate over all elements in the new tensor
  std::vector<size_t> new_indices(ndim(), 0);
  for (size_t flat_idx = 0; flat_idx < numel(); ++flat_idx) {
    // Compute old indices from new indices
    std::vector<size_t> old_indices(ndim());
    for (size_t i = 0; i < ndim(); ++i) {
      old_indices[dims_order[i]] = new_indices[i];
    }

    // Compute flat indices
    size_t old_flat = 0;
    for (size_t i = 0; i < ndim(); ++i) {
      old_flat += old_indices[i] * old_strides[i];
    }

    result.data_[flat_idx] = data_[old_flat];

    // Increment new indices
    for (int i = static_cast<int>(ndim()) - 1; i >= 0; --i) {
      new_indices[i]++;
      if (new_indices[i] < new_dims[i]) break;
      new_indices[i] = 0;
    }
  }

  return result;
}

template <typename T>
Tensor<T> Tensor<T>::squeeze(size_t dim) const {
  const auto& dims = shape_.dims();
  if (dim >= dims.size() || dims[dim] != 1) {
    throw std::runtime_error("Cannot squeeze dimension that is not 1");
  }

  std::vector<size_t> new_dims;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i != dim) new_dims.push_back(dims[i]);
  }

  if (new_dims.empty()) new_dims.push_back(1);
  return reshape(Shape(new_dims));
}

template <typename T>
Tensor<T> Tensor<T>::unsqueeze(size_t dim) const {
  const auto& dims = shape_.dims();
  if (dim > dims.size()) {
    throw std::runtime_error("Unsqueeze dimension out of range");
  }

  std::vector<size_t> new_dims;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i == dim) new_dims.push_back(1);
    new_dims.push_back(dims[i]);
  }
  if (dim == dims.size()) new_dims.push_back(1);

  return reshape(Shape(new_dims));
}

template <typename T>
Tensor<T> Tensor<T>::slice(size_t dim, size_t start, size_t end) const {
  const auto& dims = shape_.dims();
  if (dim >= dims.size()) {
    throw std::runtime_error("Slice dimension out of range");
  }
  if (start >= end || end > dims[dim]) {
    throw std::runtime_error("Invalid slice range");
  }

  std::vector<size_t> new_dims = dims;
  new_dims[dim] = end - start;
  Tensor<T> result{Shape(new_dims)};

  std::vector<size_t> strides = compute_strides();
  size_t outer_stride = (dim > 0) ? strides[dim - 1] : numel();
  size_t inner_size = (dim < dims.size() - 1) ? strides[dim] : 1;

  size_t num_outer = numel() / outer_stride;
  size_t result_idx = 0;

  for (size_t outer = 0; outer < num_outer; ++outer) {
    size_t base_idx = outer * outer_stride;
    for (size_t s = start; s < end; ++s) {
      size_t src_base = base_idx + s * inner_size;
      for (size_t inner = 0; inner < inner_size; ++inner) {
        result.data_[result_idx++] = data_[src_base + inner];
      }
    }
  }

  return result;
}

template <typename T>
Tensor<T> Tensor<T>::contiguous() const {
  return clone();
}

// Broadcasting helper
template <typename T>
template <typename Op>
Tensor<T> Tensor<T>::broadcast_op(const Tensor<T>& other, Op op) const {
  // Fast path: shapes are identical
  if (shape_ == other.shape_) {
    Tensor<T> result(shape_);
    for (size_t i = 0; i < numel(); ++i) {
      result.data_[i] = op(data_[i], other.data_[i]);
    }
    return result;
  }

  // Check if shapes can be broadcast
  if (!Shape::can_broadcast(shape_, other.shape_)) {
    throw std::runtime_error("Shapes cannot be broadcast together");
  }

  // Compute broadcast shape
  Shape result_shape = Shape::broadcast_shape(shape_, other.shape_);
  Tensor<T> result(result_shape);

  const auto& result_dims = result_shape.dims();
  const auto& a_dims = shape_.dims();
  const auto& b_dims = other.shape_.dims();
  size_t result_ndim = result_dims.size();

  // Compute strides for result tensor
  std::vector<size_t> result_strides(result_ndim);
  size_t stride = 1;
  for (int i = static_cast<int>(result_ndim) - 1; i >= 0; --i) {
    result_strides[i] = stride;
    stride *= result_dims[i];
  }

  // Compute strides for input tensors (with broadcasting: 0 for broadcast dims)
  std::vector<size_t> a_strides(result_ndim, 0);
  std::vector<size_t> b_strides(result_ndim, 0);

  size_t a_stride = 1;
  for (int i = static_cast<int>(a_dims.size()) - 1; i >= 0; --i) {
    size_t result_idx = result_ndim - a_dims.size() + i;
    if (a_dims[i] != 1) {
      a_strides[result_idx] = a_stride;
    }
    a_stride *= a_dims[i];
  }

  size_t b_stride = 1;
  for (int i = static_cast<int>(b_dims.size()) - 1; i >= 0; --i) {
    size_t result_idx = result_ndim - b_dims.size() + i;
    if (b_dims[i] != 1) {
      b_strides[result_idx] = b_stride;
    }
    b_stride *= b_dims[i];
  }

  // Iterate over result tensor
  std::vector<size_t> indices(result_ndim, 0);
  for (size_t flat_idx = 0; flat_idx < result.numel(); ++flat_idx) {
    // Compute flat indices for a and b using broadcast strides
    size_t a_flat = 0;
    size_t b_flat = 0;
    for (size_t d = 0; d < result_ndim; ++d) {
      a_flat += indices[d] * a_strides[d];
      b_flat += indices[d] * b_strides[d];
    }

    result.data_[flat_idx] = op(data_[a_flat], other.data_[b_flat]);

    // Increment indices
    for (int i = static_cast<int>(result_ndim) - 1; i >= 0; --i) {
      indices[i]++;
      if (indices[i] < result_dims[i]) break;
      indices[i] = 0;
    }
  }

  return result;
}

// Element-wise operations with broadcasting support
template <typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const {
  return broadcast_op(other, [](T a, T b) { return a + b; });
}

template <typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T>& other) const {
  return broadcast_op(other, [](T a, T b) { return a - b; });
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T>& other) const {
  return broadcast_op(other, [](T a, T b) { return a * b; });
}

template <typename T>
Tensor<T> Tensor<T>::operator/(const Tensor<T>& other) const {
  return broadcast_op(other, [](T a, T b) { return a / b; });
}

template <typename T>
Tensor<T> Tensor<T>::operator+(const T& scalar) const {
  Tensor<T> result(shape_);
  for (size_t i = 0; i < numel(); ++i) {
    result.data_[i] = data_[i] + scalar;
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator-(const T& scalar) const {
  Tensor<T> result(shape_);
  for (size_t i = 0; i < numel(); ++i) {
    result.data_[i] = data_[i] - scalar;
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const T& scalar) const {
  Tensor<T> result(shape_);
  for (size_t i = 0; i < numel(); ++i) {
    result.data_[i] = data_[i] * scalar;
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator/(const T& scalar) const {
  Tensor<T> result(shape_);
  for (size_t i = 0; i < numel(); ++i) {
    result.data_[i] = data_[i] / scalar;
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::neg() const {
  Tensor<T> result(shape_);
  for (size_t i = 0; i < numel(); ++i) {
    result.data_[i] = -data_[i];
  }
  return result;
}

// In-place operations
template <typename T>
Tensor<T>& Tensor<T>::operator+=(const Tensor<T>& other) {
  if (shape_ != other.shape_) {
    throw std::runtime_error("Tensor shapes must match for addition");
  }
  for (size_t i = 0; i < numel(); ++i) {
    data_[i] += other.data_[i];
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator-=(const Tensor<T>& other) {
  if (shape_ != other.shape_) {
    throw std::runtime_error("Tensor shapes must match for subtraction");
  }
  for (size_t i = 0; i < numel(); ++i) {
    data_[i] -= other.data_[i];
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator*=(const Tensor<T>& other) {
  if (shape_ != other.shape_) {
    throw std::runtime_error("Tensor shapes must match for multiplication");
  }
  for (size_t i = 0; i < numel(); ++i) {
    data_[i] *= other.data_[i];
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator/=(const Tensor<T>& other) {
  if (shape_ != other.shape_) {
    throw std::runtime_error("Tensor shapes must match for division");
  }
  for (size_t i = 0; i < numel(); ++i) {
    data_[i] /= other.data_[i];
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator+=(const T& scalar) {
  for (size_t i = 0; i < numel(); ++i) {
    data_[i] += scalar;
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator-=(const T& scalar) {
  for (size_t i = 0; i < numel(); ++i) {
    data_[i] -= scalar;
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator*=(const T& scalar) {
  for (size_t i = 0; i < numel(); ++i) {
    data_[i] *= scalar;
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator/=(const T& scalar) {
  for (size_t i = 0; i < numel(); ++i) {
    data_[i] /= scalar;
  }
  return *this;
}

// Activation functions
template <typename T>
Tensor<T> Tensor<T>::relu() const {
  Tensor<T> result(shape_);
  for (size_t i = 0; i < numel(); ++i) {
    result.data_[i] = data_[i] > T(0) ? data_[i] : T(0);
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::gelu() const {
  // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  const T sqrt_2_over_pi = T(0.7978845608028654);
  const T coeff = T(0.044715);
  Tensor<T> result(shape_);
  for (size_t i = 0; i < numel(); ++i) {
    T x = data_[i];
    T inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    result.data_[i] = T(0.5) * x * (T(1) + std::tanh(inner));
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::silu() const {
  // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
  Tensor<T> result(shape_);
  for (size_t i = 0; i < numel(); ++i) {
    T x = data_[i];
    result.data_[i] = x / (T(1) + std::exp(-x));
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::sigmoid() const {
  Tensor<T> result(shape_);
  for (size_t i = 0; i < numel(); ++i) {
    result.data_[i] = T(1) / (T(1) + std::exp(-data_[i]));
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::tanh_() const {
  Tensor<T> result(shape_);
  for (size_t i = 0; i < numel(); ++i) {
    result.data_[i] = std::tanh(data_[i]);
  }
  return result;
}

// Reduction operations
template <typename T>
T Tensor<T>::sum() const {
  T result = T(0);
  for (size_t i = 0; i < numel(); ++i) {
    result += data_[i];
  }
  return result;
}

template <typename T>
T Tensor<T>::mean() const {
  return sum() / static_cast<T>(numel());
}

template <typename T>
T Tensor<T>::max() const {
  if (numel() == 0) {
    throw std::runtime_error("Cannot get max of empty tensor");
  }
  T result = data_[0];
  for (size_t i = 1; i < numel(); ++i) {
    if (data_[i] > result) result = data_[i];
  }
  return result;
}

template <typename T>
T Tensor<T>::min() const {
  if (numel() == 0) {
    throw std::runtime_error("Cannot get min of empty tensor");
  }
  T result = data_[0];
  for (size_t i = 1; i < numel(); ++i) {
    if (data_[i] < result) result = data_[i];
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::sum(size_t dim, bool keepdim) const {
  const auto& dims = shape_.dims();
  if (dim >= dims.size()) {
    throw std::runtime_error("Reduction dimension out of range");
  }

  std::vector<size_t> new_dims;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i == dim) {
      if (keepdim) new_dims.push_back(1);
    } else {
      new_dims.push_back(dims[i]);
    }
  }
  if (new_dims.empty()) new_dims.push_back(1);

  Tensor<T> result(Shape(new_dims), T(0));
  std::vector<size_t> strides = compute_strides();

  // Iterate through all elements
  std::vector<size_t> indices(dims.size(), 0);
  for (size_t flat_idx = 0; flat_idx < numel(); ++flat_idx) {
    // Compute result index (skip the reduced dimension)
    std::vector<size_t> result_indices;
    for (size_t i = 0; i < dims.size(); ++i) {
      if (i == dim) {
        if (keepdim) result_indices.push_back(0);
      } else {
        result_indices.push_back(indices[i]);
      }
    }
    if (result_indices.empty()) result_indices.push_back(0);

    size_t result_flat = 0;
    size_t result_stride = 1;
    for (int i = static_cast<int>(result_indices.size()) - 1; i >= 0; --i) {
      result_flat += result_indices[i] * result_stride;
      result_stride *= new_dims[i];
    }

    result.data_[result_flat] += data_[flat_idx];

    // Increment indices
    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
      indices[i]++;
      if (indices[i] < dims[i]) break;
      indices[i] = 0;
    }
  }

  return result;
}

template <typename T>
Tensor<T> Tensor<T>::mean(size_t dim, bool keepdim) const {
  Tensor<T> result = sum(dim, keepdim);
  T count = static_cast<T>(shape_[dim]);
  for (size_t i = 0; i < result.numel(); ++i) {
    result.data_[i] /= count;
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::max(size_t dim, bool keepdim) const {
  const auto& dims = shape_.dims();
  if (dim >= dims.size()) {
    throw std::runtime_error("Reduction dimension out of range");
  }

  std::vector<size_t> new_dims;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i == dim) {
      if (keepdim) new_dims.push_back(1);
    } else {
      new_dims.push_back(dims[i]);
    }
  }
  if (new_dims.empty()) new_dims.push_back(1);

  // Initialize with very small values
  Tensor<T> result{Shape(new_dims)};
  for (size_t i = 0; i < result.numel(); ++i) {
    result.data_[i] = std::numeric_limits<T>::lowest();
  }

  std::vector<size_t> indices(dims.size(), 0);
  for (size_t flat_idx = 0; flat_idx < numel(); ++flat_idx) {
    std::vector<size_t> result_indices;
    for (size_t i = 0; i < dims.size(); ++i) {
      if (i == dim) {
        if (keepdim) result_indices.push_back(0);
      } else {
        result_indices.push_back(indices[i]);
      }
    }
    if (result_indices.empty()) result_indices.push_back(0);

    size_t result_flat = 0;
    size_t result_stride = 1;
    for (int i = static_cast<int>(result_indices.size()) - 1; i >= 0; --i) {
      result_flat += result_indices[i] * result_stride;
      result_stride *= new_dims[i];
    }

    if (data_[flat_idx] > result.data_[result_flat]) {
      result.data_[result_flat] = data_[flat_idx];
    }

    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
      indices[i]++;
      if (indices[i] < dims[i]) break;
      indices[i] = 0;
    }
  }

  return result;
}

template <typename T>
Tensor<T> Tensor<T>::min(size_t dim, bool keepdim) const {
  const auto& dims = shape_.dims();
  if (dim >= dims.size()) {
    throw std::runtime_error("Reduction dimension out of range");
  }

  std::vector<size_t> new_dims;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i == dim) {
      if (keepdim) new_dims.push_back(1);
    } else {
      new_dims.push_back(dims[i]);
    }
  }
  if (new_dims.empty()) new_dims.push_back(1);

  Tensor<T> result{Shape(new_dims)};
  for (size_t i = 0; i < result.numel(); ++i) {
    result.data_[i] = std::numeric_limits<T>::max();
  }

  std::vector<size_t> indices(dims.size(), 0);
  for (size_t flat_idx = 0; flat_idx < numel(); ++flat_idx) {
    std::vector<size_t> result_indices;
    for (size_t i = 0; i < dims.size(); ++i) {
      if (i == dim) {
        if (keepdim) result_indices.push_back(0);
      } else {
        result_indices.push_back(indices[i]);
      }
    }
    if (result_indices.empty()) result_indices.push_back(0);

    size_t result_flat = 0;
    size_t result_stride = 1;
    for (int i = static_cast<int>(result_indices.size()) - 1; i >= 0; --i) {
      result_flat += result_indices[i] * result_stride;
      result_stride *= new_dims[i];
    }

    if (data_[flat_idx] < result.data_[result_flat]) {
      result.data_[result_flat] = data_[flat_idx];
    }

    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
      indices[i]++;
      if (indices[i] < dims[i]) break;
      indices[i] = 0;
    }
  }

  return result;
}

template <typename T>
Tensor<size_t> Tensor<T>::argmax(size_t dim) const {
  const auto& dims = shape_.dims();
  if (dim >= dims.size()) {
    throw std::runtime_error("Reduction dimension out of range");
  }

  std::vector<size_t> new_dims;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i != dim) new_dims.push_back(dims[i]);
  }
  if (new_dims.empty()) new_dims.push_back(1);

  Shape result_shape(new_dims);
  Tensor<size_t> result{result_shape};
  std::vector<T> max_vals(result.numel(), std::numeric_limits<T>::lowest());

  std::vector<size_t> indices(dims.size(), 0);
  for (size_t flat_idx = 0; flat_idx < numel(); ++flat_idx) {
    std::vector<size_t> result_indices;
    for (size_t i = 0; i < dims.size(); ++i) {
      if (i != dim) result_indices.push_back(indices[i]);
    }
    if (result_indices.empty()) result_indices.push_back(0);

    size_t result_flat = 0;
    size_t result_stride = 1;
    for (int i = static_cast<int>(result_indices.size()) - 1; i >= 0; --i) {
      result_flat += result_indices[i] * result_stride;
      result_stride *= new_dims[i];
    }

    if (data_[flat_idx] > max_vals[result_flat]) {
      max_vals[result_flat] = data_[flat_idx];
      result.data()[result_flat] = indices[dim];
    }

    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
      indices[i]++;
      if (indices[i] < dims[i]) break;
      indices[i] = 0;
    }
  }

  return result;
}

template <typename T>
Tensor<size_t> Tensor<T>::argmin(size_t dim) const {
  const auto& dims = shape_.dims();
  if (dim >= dims.size()) {
    throw std::runtime_error("Reduction dimension out of range");
  }

  std::vector<size_t> new_dims;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i != dim) new_dims.push_back(dims[i]);
  }
  if (new_dims.empty()) new_dims.push_back(1);

  Shape result_shape(new_dims);
  Tensor<size_t> result{result_shape};
  std::vector<T> min_vals(result.numel(), std::numeric_limits<T>::max());

  std::vector<size_t> indices(dims.size(), 0);
  for (size_t flat_idx = 0; flat_idx < numel(); ++flat_idx) {
    std::vector<size_t> result_indices;
    for (size_t i = 0; i < dims.size(); ++i) {
      if (i != dim) result_indices.push_back(indices[i]);
    }
    if (result_indices.empty()) result_indices.push_back(0);

    size_t result_flat = 0;
    size_t result_stride = 1;
    for (int i = static_cast<int>(result_indices.size()) - 1; i >= 0; --i) {
      result_flat += result_indices[i] * result_stride;
      result_stride *= new_dims[i];
    }

    if (data_[flat_idx] < min_vals[result_flat]) {
      min_vals[result_flat] = data_[flat_idx];
      result.data()[result_flat] = indices[dim];
    }

    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
      indices[i]++;
      if (indices[i] < dims[i]) break;
      indices[i] = 0;
    }
  }

  return result;
}

// Matrix multiplication with broadcasting support for batch dimensions
// Supports GQA (Grouped-Query Attention) where num_q_heads > num_kv_heads
template <typename T>
Tensor<T> Tensor<T>::matmul(const Tensor<T>& other) const {
  // Handle 2D matrix multiplication
  if (ndim() == 2 && other.ndim() == 2) {
    size_t m = shape_[0];
    size_t k = shape_[1];
    size_t n = other.shape_[1];

    if (k != other.shape_[0]) {
      throw std::runtime_error(
          "Matrix dimensions don't match for multiplication");
    }

    Tensor<T> result(Shape({m, n}), T(0));

    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        T sum = T(0);
        for (size_t l = 0; l < k; ++l) {
          sum += (*this)(i, l) * other(l, j);
        }
        result(i, j) = sum;
      }
    }
    return result;
  }

  // Handle batched matrix multiplication with broadcasting
  // (batch1, m, k) x (batch2, k, n) -> (max(batch1,batch2), m, n)
  if (ndim() == 3 && other.ndim() == 3) {
    size_t batch1 = shape_[0];
    size_t batch2 = other.shape_[0];
    size_t m = shape_[1];
    size_t k = shape_[2];
    size_t n = other.shape_[2];

    if (k != other.shape_[1]) {
      throw std::runtime_error(
          "Batched matrix dimensions don't match for multiplication");
    }

    // Check batch dimension broadcasting
    if (batch1 != batch2 && batch1 != 1 && batch2 != 1) {
      throw std::runtime_error(
          "Batch dimensions are not broadcastable");
    }

    size_t batch_out = std::max(batch1, batch2);
    Tensor<T> result(Shape({batch_out, m, n}), T(0));

    for (size_t b = 0; b < batch_out; ++b) {
      size_t b1 = (batch1 == 1) ? 0 : b;
      size_t b2 = (batch2 == 1) ? 0 : b;
      for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
          T sum = T(0);
          for (size_t l = 0; l < k; ++l) {
            sum += (*this)(b1, i, l) * other(b2, l, j);
          }
          result(b, i, j) = sum;
        }
      }
    }
    return result;
  }

  // Handle 4D batched matmul with broadcasting for GQA
  // (batch1, heads1, seq, dim) x (batch2, heads2, dim, k)
  // Supports GQA where heads1 > heads2 and heads1 % heads2 == 0
  if (ndim() == 4 && other.ndim() == 4) {
    size_t batch1 = shape_[0];
    size_t batch2 = other.shape_[0];
    size_t heads1 = shape_[1];
    size_t heads2 = other.shape_[1];
    size_t m = shape_[2];
    size_t k = shape_[3];
    size_t n = other.shape_[3];

    if (k != other.shape_[2]) {
      throw std::runtime_error(
          "4D matrix dimensions don't match for multiplication");
    }

    // Check batch dimension broadcasting
    if (batch1 != batch2 && batch1 != 1 && batch2 != 1) {
      throw std::runtime_error(
          "Batch dimensions are not broadcastable");
    }

    // Check heads dimension broadcasting (for GQA support)
    // Allow: same heads, one is 1, or heads1 is multiple of heads2 (GQA)
    bool heads_broadcastable = (heads1 == heads2) ||
                                (heads1 == 1) ||
                                (heads2 == 1) ||
                                (heads1 > heads2 && heads1 % heads2 == 0) ||
                                (heads2 > heads1 && heads2 % heads1 == 0);
    if (!heads_broadcastable) {
      throw std::runtime_error(
          "Head dimensions are not broadcastable (for GQA, num_q_heads must be divisible by num_kv_heads)");
    }

    size_t batch_out = std::max(batch1, batch2);
    size_t heads_out = std::max(heads1, heads2);
    Tensor<T> result(Shape({batch_out, heads_out, m, n}), T(0));

    for (size_t b = 0; b < batch_out; ++b) {
      size_t b1 = (batch1 == 1) ? 0 : b;
      size_t b2 = (batch2 == 1) ? 0 : b;
      for (size_t h = 0; h < heads_out; ++h) {
        // For GQA: map query head to corresponding KV head
        size_t h1, h2;
        if (heads1 == heads2) {
          h1 = h;
          h2 = h;
        } else if (heads1 == 1) {
          h1 = 0;
          h2 = h;
        } else if (heads2 == 1) {
          h1 = h;
          h2 = 0;
        } else if (heads1 > heads2) {
          // GQA: multiple query heads per KV head
          h1 = h;
          h2 = h / (heads1 / heads2);
        } else {
          // Reverse GQA (unusual but supported)
          h1 = h / (heads2 / heads1);
          h2 = h;
        }

        for (size_t i = 0; i < m; ++i) {
          for (size_t j = 0; j < n; ++j) {
            T sum = T(0);
            for (size_t l = 0; l < k; ++l) {
              sum += (*this)(b1, h1, i, l) * other(b2, h2, l, j);
            }
            result(b, h, i, j) = sum;
          }
        }
      }
    }
    return result;
  }

  throw std::runtime_error("matmul only supports 2D, 3D, and 4D tensors");
}

// Normalization operations
template <typename T>
Tensor<T> Tensor<T>::softmax(size_t dim) const {
  const auto& dims = shape_.dims();
  if (dim >= dims.size()) {
    throw std::runtime_error("Softmax dimension out of range");
  }

  // Compute max along dimension for numerical stability
  Tensor<T> max_vals = max(dim, true);
  Tensor<T> result(shape_);

  // Compute exp(x - max) and sum
  std::vector<size_t> strides = compute_strides();

  // Copy data and subtract max, then exp
  std::vector<size_t> indices(dims.size(), 0);
  for (size_t flat_idx = 0; flat_idx < numel(); ++flat_idx) {
    // Get max value for this position
    std::vector<size_t> max_indices;
    for (size_t i = 0; i < dims.size(); ++i) {
      if (i == dim) {
        max_indices.push_back(0);
      } else {
        max_indices.push_back(indices[i]);
      }
    }
    T max_val = max_vals.at(max_indices);
    result.data_[flat_idx] = std::exp(data_[flat_idx] - max_val);

    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
      indices[i]++;
      if (indices[i] < dims[i]) break;
      indices[i] = 0;
    }
  }

  // Compute sum along dimension and divide
  Tensor<T> sum_vals = result.sum(dim, true);

  std::fill(indices.begin(), indices.end(), 0);
  for (size_t flat_idx = 0; flat_idx < numel(); ++flat_idx) {
    std::vector<size_t> sum_indices;
    for (size_t i = 0; i < dims.size(); ++i) {
      if (i == dim) {
        sum_indices.push_back(0);
      } else {
        sum_indices.push_back(indices[i]);
      }
    }
    result.data_[flat_idx] /= sum_vals.at(sum_indices);

    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
      indices[i]++;
      if (indices[i] < dims[i]) break;
      indices[i] = 0;
    }
  }

  return result;
}

template <typename T>
Tensor<T> Tensor<T>::layer_norm(const Tensor<T>& weight, const Tensor<T>& bias,
                                T eps) const {
  // Layer norm normalizes over the last dimension
  if (ndim() < 1) {
    throw std::runtime_error("Layer norm requires at least 1D tensor");
  }

  size_t norm_dim = ndim() - 1;
  size_t norm_size = shape_[norm_dim];

  if (weight.numel() != norm_size || bias.numel() != norm_size) {
    throw std::runtime_error("Weight and bias size must match last dimension");
  }

  Tensor<T> result(shape_);

  // Compute mean along last dimension
  Tensor<T> mean_vals = mean(norm_dim, true);

  // Compute variance: E[(x - mean)^2]
  // We need to compute this without broadcasting, so do it element by element
  Tensor<T> var_vals = Tensor<T>::zeros(mean_vals.shape());

  std::vector<size_t> indices(ndim(), 0);
  for (size_t flat_idx = 0; flat_idx < numel(); ++flat_idx) {
    std::vector<size_t> mean_indices;
    for (size_t i = 0; i < ndim(); ++i) {
      if (i == norm_dim) {
        mean_indices.push_back(0);
      } else {
        mean_indices.push_back(indices[i]);
      }
    }

    T mean_val = mean_vals.at(mean_indices);
    T diff = data_[flat_idx] - mean_val;
    var_vals.at(mean_indices) += diff * diff;

    for (int i = static_cast<int>(ndim()) - 1; i >= 0; --i) {
      indices[i]++;
      if (indices[i] < shape_[i]) break;
      indices[i] = 0;
    }
  }

  // Divide by norm_size to get variance
  for (size_t i = 0; i < var_vals.numel(); ++i) {
    var_vals.data()[i] /= static_cast<T>(norm_size);
  }

  // Normalize: (x - mean) / sqrt(var + eps) * weight + bias
  std::fill(indices.begin(), indices.end(), 0);
  for (size_t flat_idx = 0; flat_idx < numel(); ++flat_idx) {
    std::vector<size_t> stat_indices;
    for (size_t i = 0; i < ndim(); ++i) {
      if (i == norm_dim) {
        stat_indices.push_back(0);
      } else {
        stat_indices.push_back(indices[i]);
      }
    }

    T var_val = var_vals.at(stat_indices);
    T mean_val = mean_vals.at(stat_indices);
    size_t weight_idx = indices[norm_dim];

    result.data_[flat_idx] = (data_[flat_idx] - mean_val) /
                                 std::sqrt(var_val + eps) *
                                 weight.data()[weight_idx] +
                             bias.data()[weight_idx];

    for (int i = static_cast<int>(ndim()) - 1; i >= 0; --i) {
      indices[i]++;
      if (indices[i] < shape_[i]) break;
      indices[i] = 0;
    }
  }

  return result;
}

template <typename T>
Tensor<T> Tensor<T>::rms_norm(const Tensor<T>& weight, T eps) const {
  // RMS Norm: x / sqrt(mean(x^2) + eps) * weight
  if (ndim() < 1) {
    throw std::runtime_error("RMS norm requires at least 1D tensor");
  }

  size_t norm_dim = ndim() - 1;
  size_t norm_size = shape_[norm_dim];

  if (weight.numel() != norm_size) {
    throw std::runtime_error("Weight size must match last dimension");
  }

  Tensor<T> result(shape_);

  // Compute mean of squares along last dimension
  Tensor<T> sq = (*this) * (*this);
  Tensor<T> mean_sq = sq.mean(norm_dim, true);

  // Normalize
  std::vector<size_t> indices(ndim(), 0);
  for (size_t flat_idx = 0; flat_idx < numel(); ++flat_idx) {
    std::vector<size_t> rms_indices;
    for (size_t i = 0; i < ndim(); ++i) {
      if (i == norm_dim) {
        rms_indices.push_back(0);
      } else {
        rms_indices.push_back(indices[i]);
      }
    }

    T mean_sq_val = mean_sq.at(rms_indices);
    T rms = std::sqrt(mean_sq_val + eps);
    size_t weight_idx = indices[norm_dim];

    result.data_[flat_idx] = data_[flat_idx] / rms * weight.data()[weight_idx];

    for (int i = static_cast<int>(ndim()) - 1; i >= 0; --i) {
      indices[i]++;
      if (indices[i] < shape_[i]) break;
      indices[i] = 0;
    }
  }

  return result;
}

// Utility functions
template <typename T>
void Tensor<T>::fill(const T& value) {
  std::fill(data_.begin(), data_.end(), value);
}

template <typename T>
Tensor<T> Tensor<T>::clone() const {
  Tensor<T> result(shape_);
  std::copy(data_.begin(), data_.end(), result.data_.begin());
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::zeros(const Shape& s) {
  return Tensor<T>(s, T(0));
}

template <typename T>
Tensor<T> Tensor<T>::ones(const Shape& s) {
  return Tensor<T>(s, T(1));
}

template <typename T>
Tensor<T> Tensor<T>::full(const Shape& s, const T& value) {
  return Tensor<T>(s, value);
}

template <typename T>
Tensor<T> Tensor<T>::apply(std::function<T(T)> func) const {
  Tensor<T> result(shape_);
  for (size_t i = 0; i < numel(); ++i) {
    result.data_[i] = func(data_[i]);
  }
  return result;
}

// Free function operators for scalar on left side
template <typename T>
Tensor<T> operator+(const T& scalar, const Tensor<T>& tensor) {
  return tensor + scalar;
}

template <typename T>
Tensor<T> operator-(const T& scalar, const Tensor<T>& tensor) {
  Tensor<T> result(tensor.shape());
  for (size_t i = 0; i < tensor.numel(); ++i) {
    result.data()[i] = scalar - tensor.data()[i];
  }
  return result;
}

template <typename T>
Tensor<T> operator*(const T& scalar, const Tensor<T>& tensor) {
  return tensor * scalar;
}

template <typename T>
Tensor<T> operator/(const T& scalar, const Tensor<T>& tensor) {
  Tensor<T> result(tensor.shape());
  for (size_t i = 0; i < tensor.numel(); ++i) {
    result.data()[i] = scalar / tensor.data()[i];
  }
  return result;
}
