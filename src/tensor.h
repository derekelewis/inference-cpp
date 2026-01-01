#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

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
  static Tensor<T> concat(const std::vector<Tensor<T>>& tensors, size_t dim);

  // Apply function element-wise
  Tensor<T> apply(std::function<T(T)> func) const;

  // Rotary Position Embedding (RoPE)
  // Expects shape [..., seq_len, head_dim] where head_dim is even
  // start_pos: starting position for the sequence (for incremental decoding)
  // base: RoPE base frequency (default 10000)
  Tensor<T> rope(size_t start_pos, T base = T(10000)) const;

  // Create causal attention mask
  // Returns [seq_len, total_len] mask where mask[i,j] = -inf if j > i + offset
  // offset = total_len - seq_len (for KV cache scenarios)
  static Tensor<T> causal_mask(size_t seq_len, size_t total_len);
  static Tensor<T> causal_mask(size_t seq_len) { return causal_mask(seq_len, seq_len); }

 public:
  // Check if tensor data is contiguous in memory
  bool is_contiguous() const;

 private:
  Shape shape_;
  std::shared_ptr<std::vector<T>> storage_;  // Shared underlying data
  size_t offset_ = 0;                         // Start offset into storage
  std::vector<size_t> strides_;               // Element strides per dimension

  // Private constructor for creating views
  Tensor(std::shared_ptr<std::vector<T>> storage, size_t offset,
         const Shape& shape, std::vector<size_t> strides);

  // Helper functions
  size_t compute_flat_index(const std::vector<size_t>& indices) const;
  static std::vector<size_t> compute_contiguous_strides(const Shape& shape);

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
Tensor<T>::Tensor()
    : shape_({1})
    , storage_(std::make_shared<std::vector<T>>(1))
    , offset_(0)
    , strides_(compute_contiguous_strides(shape_)) {}

template <typename T>
Tensor<T>::Tensor(const Shape& s)
    : shape_(s)
    , storage_(std::make_shared<std::vector<T>>(s.numel()))
    , offset_(0)
    , strides_(compute_contiguous_strides(s)) {}

template <typename T>
Tensor<T>::Tensor(const Shape& s, const T& fill_value)
    : shape_(s)
    , storage_(std::make_shared<std::vector<T>>(s.numel(), fill_value))
    , offset_(0)
    , strides_(compute_contiguous_strides(s)) {}

template <typename T>
Tensor<T>::Tensor(const Shape& s, const std::vector<T>& data)
    : shape_(s)
    , storage_(std::make_shared<std::vector<T>>(data))
    , offset_(0)
    , strides_(compute_contiguous_strides(s)) {
  if (storage_->size() != s.numel()) {
    throw std::runtime_error("Data size does not match shape size");
  }
}

// Private view constructor - shares storage with another tensor
template <typename T>
Tensor<T>::Tensor(std::shared_ptr<std::vector<T>> storage, size_t offset,
                  const Shape& shape, std::vector<size_t> strides)
    : shape_(shape)
    , storage_(storage)
    , offset_(offset)
    , strides_(std::move(strides)) {}

template <typename T>
Tensor<T>::Tensor(const Tensor& other)
    : shape_(other.shape_)
    , storage_(other.storage_)
    , offset_(other.offset_)
    , strides_(other.strides_) {}

template <typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_))
    , storage_(std::move(other.storage_))
    , offset_(other.offset_)
    , strides_(std::move(other.strides_)) {}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& other) {
  if (this != &other) {
    shape_ = other.shape_;
    storage_ = other.storage_;
    offset_ = other.offset_;
    strides_ = other.strides_;
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    shape_ = std::move(other.shape_);
    storage_ = std::move(other.storage_);
    offset_ = other.offset_;
    strides_ = std::move(other.strides_);
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
  return shape_.numel();
}

template <typename T>
size_t Tensor<T>::size(size_t dim) const {
  return shape_[dim];
}

// Data access
template <typename T>
T* Tensor<T>::data() {
  return storage_->data() + offset_;
}

template <typename T>
const T* Tensor<T>::data() const {
  return storage_->data() + offset_;
}

template <typename T>
T& Tensor<T>::operator()(size_t i) {
  // Flat indexing - assumes contiguous layout or 1D tensor
  return data()[i];
}

template <typename T>
const T& Tensor<T>::operator()(size_t i) const {
  // Flat indexing - assumes contiguous layout or 1D tensor
  return data()[i];
}

template <typename T>
T& Tensor<T>::operator()(size_t i, size_t j) {
  return (*storage_)[offset_ + i * strides_[0] + j * strides_[1]];
}

template <typename T>
const T& Tensor<T>::operator()(size_t i, size_t j) const {
  return (*storage_)[offset_ + i * strides_[0] + j * strides_[1]];
}

template <typename T>
T& Tensor<T>::operator()(size_t i, size_t j, size_t k) {
  return (*storage_)[offset_ + i * strides_[0] + j * strides_[1] + k * strides_[2]];
}

template <typename T>
const T& Tensor<T>::operator()(size_t i, size_t j, size_t k) const {
  return (*storage_)[offset_ + i * strides_[0] + j * strides_[1] + k * strides_[2]];
}

template <typename T>
T& Tensor<T>::operator()(size_t i, size_t j, size_t k, size_t l) {
  return (*storage_)[offset_ + i * strides_[0] + j * strides_[1] + k * strides_[2] + l * strides_[3]];
}

template <typename T>
const T& Tensor<T>::operator()(size_t i, size_t j, size_t k, size_t l) const {
  return (*storage_)[offset_ + i * strides_[0] + j * strides_[1] + k * strides_[2] + l * strides_[3]];
}

template <typename T>
size_t Tensor<T>::compute_flat_index(const std::vector<size_t>& indices) const {
  if (indices.size() != strides_.size()) {
    throw std::runtime_error("Index dimensions don't match tensor dimensions");
  }
  size_t flat_idx = offset_;
  for (size_t i = 0; i < indices.size(); ++i) {
    flat_idx += indices[i] * strides_[i];
  }
  return flat_idx;
}

template <typename T>
std::vector<size_t> Tensor<T>::compute_contiguous_strides(const Shape& shape) {
  const auto& dims = shape.dims();
  std::vector<size_t> strides(dims.size());
  size_t stride = 1;
  for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= dims[i];
  }
  return strides;
}

template <typename T>
bool Tensor<T>::is_contiguous() const {
  size_t expected = 1;
  for (int i = static_cast<int>(ndim()) - 1; i >= 0; --i) {
    if (strides_[i] != expected) return false;
    expected *= shape_[i];
  }
  return true;
}

template <typename T>
T& Tensor<T>::at(const std::vector<size_t>& indices) {
  return (*storage_)[compute_flat_index(indices)];
}

template <typename T>
const T& Tensor<T>::at(const std::vector<size_t>& indices) const {
  return (*storage_)[compute_flat_index(indices)];
}

// Shape manipulation
template <typename T>
Tensor<T> Tensor<T>::reshape(const Shape& new_shape) const {
  if (new_shape.numel() != numel()) {
    throw std::runtime_error(
        "Cannot reshape: total number of elements must remain the same");
  }
  // Can only create view if contiguous - otherwise need to copy first
  if (!is_contiguous()) {
    return contiguous().reshape(new_shape);
  }
  // Return a view with same storage, new shape and contiguous strides
  return Tensor<T>(storage_, offset_, new_shape,
                   compute_contiguous_strides(new_shape));
}

template <typename T>
Tensor<T> Tensor<T>::transpose() const {
  if (ndim() != 2) {
    throw std::runtime_error("transpose() without args requires 2D tensor");
  }
  // Return view with swapped shape and strides
  Shape new_shape({shape_[1], shape_[0]});
  std::vector<size_t> new_strides = {strides_[1], strides_[0]};
  return Tensor<T>(storage_, offset_, new_shape, new_strides);
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

  // Build new shape and strides by reordering according to dims_order
  const auto& old_dims = shape_.dims();
  std::vector<size_t> new_dims(ndim());
  std::vector<size_t> new_strides(ndim());
  for (size_t i = 0; i < ndim(); ++i) {
    new_dims[i] = old_dims[dims_order[i]];
    new_strides[i] = strides_[dims_order[i]];
  }

  // Return view with permuted shape and strides
  return Tensor<T>(storage_, offset_, Shape(new_dims), new_strides);
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

  // Create new shape with sliced dimension
  std::vector<size_t> new_dims = dims;
  new_dims[dim] = end - start;

  // Compute new offset: move start * stride[dim] elements into the storage
  size_t new_offset = offset_ + start * strides_[dim];

  // Return view with adjusted offset and shape (strides stay the same)
  return Tensor<T>(storage_, new_offset, Shape(new_dims), strides_);
}

template <typename T>
Tensor<T> Tensor<T>::contiguous() const {
  if (is_contiguous()) {
    return *this;  // Already contiguous, return shallow copy (view)
  }
  // Create new contiguous storage and copy data
  Tensor<T> result(shape_);
  // Copy with stride-aware iteration
  const auto& dims = shape_.dims();
  std::vector<size_t> indices(ndim(), 0);
  for (size_t flat_idx = 0; flat_idx < numel(); ++flat_idx) {
    (*result.storage_)[flat_idx] = at(indices);
    // Increment indices
    for (int i = static_cast<int>(ndim()) - 1; i >= 0; --i) {
      indices[i]++;
      if (indices[i] < dims[i]) break;
      indices[i] = 0;
    }
  }
  return result;
}

// Broadcasting helper
template <typename T>
template <typename Op>
Tensor<T> Tensor<T>::broadcast_op(const Tensor<T>& other, Op op) const {
  // Fast path: shapes are identical and both contiguous
  if (shape_ == other.shape_ && is_contiguous() && other.is_contiguous()) {
    Tensor<T> result(shape_);
    const T* a_data = data();
    const T* b_data = other.data();
    T* r_data = result.data();
    for (size_t i = 0; i < numel(); ++i) {
      r_data[i] = op(a_data[i], b_data[i]);
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

  // Compute broadcast strides for input tensors
  // For each dimension in result, compute what stride to use for a and b
  // Use 0 for broadcast dimensions (size 1) or missing dimensions
  std::vector<size_t> a_bcast_strides(result_ndim, 0);
  std::vector<size_t> b_bcast_strides(result_ndim, 0);

  for (size_t i = 0; i < result_ndim; ++i) {
    int a_idx = static_cast<int>(i) - (static_cast<int>(result_ndim) - static_cast<int>(a_dims.size()));
    int b_idx = static_cast<int>(i) - (static_cast<int>(result_ndim) - static_cast<int>(b_dims.size()));

    if (a_idx >= 0 && a_dims[a_idx] != 1) {
      a_bcast_strides[i] = strides_[a_idx];
    }
    if (b_idx >= 0 && b_dims[b_idx] != 1) {
      b_bcast_strides[i] = other.strides_[b_idx];
    }
  }

  // Iterate over result tensor
  T* r_data = result.data();
  std::vector<size_t> indices(result_ndim, 0);
  for (size_t flat_idx = 0; flat_idx < result.numel(); ++flat_idx) {
    // Compute offsets for a and b using broadcast strides
    size_t a_offset = offset_;
    size_t b_offset = other.offset_;
    for (size_t d = 0; d < result_ndim; ++d) {
      a_offset += indices[d] * a_bcast_strides[d];
      b_offset += indices[d] * b_bcast_strides[d];
    }

    r_data[flat_idx] = op((*storage_)[a_offset], (*other.storage_)[b_offset]);

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
  T* r_data = result.data();
  if (is_contiguous()) {
    const T* src = data();
    for (size_t i = 0; i < numel(); ++i) {
      r_data[i] = src[i] + scalar;
    }
  } else {
    const auto& dims = shape_.dims();
    std::vector<size_t> indices(ndim(), 0);
    for (size_t i = 0; i < numel(); ++i) {
      r_data[i] = at(indices) + scalar;
      for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
        if (++indices[d] < dims[d]) break;
        indices[d] = 0;
      }
    }
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator-(const T& scalar) const {
  Tensor<T> result(shape_);
  T* r_data = result.data();
  if (is_contiguous()) {
    const T* src = data();
    for (size_t i = 0; i < numel(); ++i) {
      r_data[i] = src[i] - scalar;
    }
  } else {
    const auto& dims = shape_.dims();
    std::vector<size_t> indices(ndim(), 0);
    for (size_t i = 0; i < numel(); ++i) {
      r_data[i] = at(indices) - scalar;
      for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
        if (++indices[d] < dims[d]) break;
        indices[d] = 0;
      }
    }
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const T& scalar) const {
  Tensor<T> result(shape_);
  T* r_data = result.data();
  if (is_contiguous()) {
    const T* src = data();
    for (size_t i = 0; i < numel(); ++i) {
      r_data[i] = src[i] * scalar;
    }
  } else {
    const auto& dims = shape_.dims();
    std::vector<size_t> indices(ndim(), 0);
    for (size_t i = 0; i < numel(); ++i) {
      r_data[i] = at(indices) * scalar;
      for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
        if (++indices[d] < dims[d]) break;
        indices[d] = 0;
      }
    }
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator/(const T& scalar) const {
  Tensor<T> result(shape_);
  T* r_data = result.data();
  if (is_contiguous()) {
    const T* src = data();
    for (size_t i = 0; i < numel(); ++i) {
      r_data[i] = src[i] / scalar;
    }
  } else {
    const auto& dims = shape_.dims();
    std::vector<size_t> indices(ndim(), 0);
    for (size_t i = 0; i < numel(); ++i) {
      r_data[i] = at(indices) / scalar;
      for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
        if (++indices[d] < dims[d]) break;
        indices[d] = 0;
      }
    }
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::neg() const {
  Tensor<T> result(shape_);
  T* r_data = result.data();
  if (is_contiguous()) {
    const T* src = data();
    for (size_t i = 0; i < numel(); ++i) {
      r_data[i] = -src[i];
    }
  } else {
    const auto& dims = shape_.dims();
    std::vector<size_t> indices(ndim(), 0);
    for (size_t i = 0; i < numel(); ++i) {
      r_data[i] = -at(indices);
      for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
        if (++indices[d] < dims[d]) break;
        indices[d] = 0;
      }
    }
  }
  return result;
}

// In-place operations
template <typename T>
Tensor<T>& Tensor<T>::operator+=(const Tensor<T>& other) {
  if (shape_ != other.shape_) {
    throw std::runtime_error("Tensor shapes must match for addition");
  }
  const auto& dims = shape_.dims();
  std::vector<size_t> indices(ndim(), 0);
  for (size_t i = 0; i < numel(); ++i) {
    at(indices) += other.at(indices);
    for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
      if (++indices[d] < dims[d]) break;
      indices[d] = 0;
    }
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator-=(const Tensor<T>& other) {
  if (shape_ != other.shape_) {
    throw std::runtime_error("Tensor shapes must match for subtraction");
  }
  const auto& dims = shape_.dims();
  std::vector<size_t> indices(ndim(), 0);
  for (size_t i = 0; i < numel(); ++i) {
    at(indices) -= other.at(indices);
    for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
      if (++indices[d] < dims[d]) break;
      indices[d] = 0;
    }
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator*=(const Tensor<T>& other) {
  if (shape_ != other.shape_) {
    throw std::runtime_error("Tensor shapes must match for multiplication");
  }
  const auto& dims = shape_.dims();
  std::vector<size_t> indices(ndim(), 0);
  for (size_t i = 0; i < numel(); ++i) {
    at(indices) *= other.at(indices);
    for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
      if (++indices[d] < dims[d]) break;
      indices[d] = 0;
    }
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator/=(const Tensor<T>& other) {
  if (shape_ != other.shape_) {
    throw std::runtime_error("Tensor shapes must match for division");
  }
  const auto& dims = shape_.dims();
  std::vector<size_t> indices(ndim(), 0);
  for (size_t i = 0; i < numel(); ++i) {
    at(indices) /= other.at(indices);
    for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
      if (++indices[d] < dims[d]) break;
      indices[d] = 0;
    }
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator+=(const T& scalar) {
  const auto& dims = shape_.dims();
  std::vector<size_t> indices(ndim(), 0);
  for (size_t i = 0; i < numel(); ++i) {
    at(indices) += scalar;
    for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
      if (++indices[d] < dims[d]) break;
      indices[d] = 0;
    }
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator-=(const T& scalar) {
  const auto& dims = shape_.dims();
  std::vector<size_t> indices(ndim(), 0);
  for (size_t i = 0; i < numel(); ++i) {
    at(indices) -= scalar;
    for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
      if (++indices[d] < dims[d]) break;
      indices[d] = 0;
    }
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator*=(const T& scalar) {
  const auto& dims = shape_.dims();
  std::vector<size_t> indices(ndim(), 0);
  for (size_t i = 0; i < numel(); ++i) {
    at(indices) *= scalar;
    for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
      if (++indices[d] < dims[d]) break;
      indices[d] = 0;
    }
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator/=(const T& scalar) {
  const auto& dims = shape_.dims();
  std::vector<size_t> indices(ndim(), 0);
  for (size_t i = 0; i < numel(); ++i) {
    at(indices) /= scalar;
    for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
      if (++indices[d] < dims[d]) break;
      indices[d] = 0;
    }
  }
  return *this;
}

// Activation functions
template <typename T>
Tensor<T> Tensor<T>::relu() const {
  Tensor<T> result(shape_);
  T* r_data = result.data();
  if (is_contiguous()) {
    const T* src = data();
    for (size_t i = 0; i < numel(); ++i) {
      r_data[i] = src[i] > T(0) ? src[i] : T(0);
    }
  } else {
    const auto& dims = shape_.dims();
    std::vector<size_t> indices(ndim(), 0);
    for (size_t i = 0; i < numel(); ++i) {
      T v = at(indices);
      r_data[i] = v > T(0) ? v : T(0);
      for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
        if (++indices[d] < dims[d]) break;
        indices[d] = 0;
      }
    }
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::gelu() const {
  // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  const T sqrt_2_over_pi = T(0.7978845608028654);
  const T coeff = T(0.044715);
  Tensor<T> result(shape_);
  T* r_data = result.data();
  if (is_contiguous()) {
    const T* src = data();
    for (size_t i = 0; i < numel(); ++i) {
      T x = src[i];
      T inner = sqrt_2_over_pi * (x + coeff * x * x * x);
      r_data[i] = T(0.5) * x * (T(1) + std::tanh(inner));
    }
  } else {
    const auto& dims = shape_.dims();
    std::vector<size_t> indices(ndim(), 0);
    for (size_t i = 0; i < numel(); ++i) {
      T x = at(indices);
      T inner = sqrt_2_over_pi * (x + coeff * x * x * x);
      r_data[i] = T(0.5) * x * (T(1) + std::tanh(inner));
      for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
        if (++indices[d] < dims[d]) break;
        indices[d] = 0;
      }
    }
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::silu() const {
  // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
  Tensor<T> result(shape_);
  T* r_data = result.data();
  if (is_contiguous()) {
    const T* src = data();
    for (size_t i = 0; i < numel(); ++i) {
      T x = src[i];
      r_data[i] = x / (T(1) + std::exp(-x));
    }
  } else {
    const auto& dims = shape_.dims();
    std::vector<size_t> indices(ndim(), 0);
    for (size_t i = 0; i < numel(); ++i) {
      T x = at(indices);
      r_data[i] = x / (T(1) + std::exp(-x));
      for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
        if (++indices[d] < dims[d]) break;
        indices[d] = 0;
      }
    }
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::sigmoid() const {
  Tensor<T> result(shape_);
  T* r_data = result.data();
  if (is_contiguous()) {
    const T* src = data();
    for (size_t i = 0; i < numel(); ++i) {
      r_data[i] = T(1) / (T(1) + std::exp(-src[i]));
    }
  } else {
    const auto& dims = shape_.dims();
    std::vector<size_t> indices(ndim(), 0);
    for (size_t i = 0; i < numel(); ++i) {
      r_data[i] = T(1) / (T(1) + std::exp(-at(indices)));
      for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
        if (++indices[d] < dims[d]) break;
        indices[d] = 0;
      }
    }
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::tanh_() const {
  Tensor<T> result(shape_);
  T* r_data = result.data();
  if (is_contiguous()) {
    const T* src = data();
    for (size_t i = 0; i < numel(); ++i) {
      r_data[i] = std::tanh(src[i]);
    }
  } else {
    const auto& dims = shape_.dims();
    std::vector<size_t> indices(ndim(), 0);
    for (size_t i = 0; i < numel(); ++i) {
      r_data[i] = std::tanh(at(indices));
      for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
        if (++indices[d] < dims[d]) break;
        indices[d] = 0;
      }
    }
  }
  return result;
}

// Reduction operations
template <typename T>
T Tensor<T>::sum() const {
  T result = T(0);
  if (is_contiguous()) {
    const T* src = data();
    for (size_t i = 0; i < numel(); ++i) {
      result += src[i];
    }
  } else {
    const auto& dims = shape_.dims();
    std::vector<size_t> indices(ndim(), 0);
    for (size_t i = 0; i < numel(); ++i) {
      result += at(indices);
      for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
        if (++indices[d] < dims[d]) break;
        indices[d] = 0;
      }
    }
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
  const auto& dims = shape_.dims();
  std::vector<size_t> indices(ndim(), 0);
  T result = at(indices);
  // Increment to first position
  for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
    if (++indices[d] < dims[d]) break;
    indices[d] = 0;
  }
  for (size_t i = 1; i < numel(); ++i) {
    T v = at(indices);
    if (v > result) result = v;
    for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
      if (++indices[d] < dims[d]) break;
      indices[d] = 0;
    }
  }
  return result;
}

template <typename T>
T Tensor<T>::min() const {
  if (numel() == 0) {
    throw std::runtime_error("Cannot get min of empty tensor");
  }
  const auto& dims = shape_.dims();
  std::vector<size_t> indices(ndim(), 0);
  T result = at(indices);
  // Increment to first position
  for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
    if (++indices[d] < dims[d]) break;
    indices[d] = 0;
  }
  for (size_t i = 1; i < numel(); ++i) {
    T v = at(indices);
    if (v < result) result = v;
    for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
      if (++indices[d] < dims[d]) break;
      indices[d] = 0;
    }
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
  T* r_data = result.data();

  // Iterate through all elements using strided access
  std::vector<size_t> indices(dims.size(), 0);
  for (size_t i = 0; i < numel(); ++i) {
    // Compute result index (skip the reduced dimension)
    std::vector<size_t> result_indices;
    for (size_t d = 0; d < dims.size(); ++d) {
      if (d == dim) {
        if (keepdim) result_indices.push_back(0);
      } else {
        result_indices.push_back(indices[d]);
      }
    }
    if (result_indices.empty()) result_indices.push_back(0);

    size_t result_flat = 0;
    size_t result_stride = 1;
    for (int d = static_cast<int>(result_indices.size()) - 1; d >= 0; --d) {
      result_flat += result_indices[d] * result_stride;
      result_stride *= new_dims[d];
    }

    r_data[result_flat] += at(indices);

    // Increment indices
    for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d) {
      if (++indices[d] < dims[d]) break;
      indices[d] = 0;
    }
  }

  return result;
}

template <typename T>
Tensor<T> Tensor<T>::mean(size_t dim, bool keepdim) const {
  Tensor<T> result = sum(dim, keepdim);
  T count = static_cast<T>(shape_[dim]);
  T* r_data = result.data();
  for (size_t i = 0; i < result.numel(); ++i) {
    r_data[i] /= count;
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
  T* r_data = result.data();
  for (size_t i = 0; i < result.numel(); ++i) {
    r_data[i] = std::numeric_limits<T>::lowest();
  }

  std::vector<size_t> indices(dims.size(), 0);
  for (size_t i = 0; i < numel(); ++i) {
    std::vector<size_t> result_indices;
    for (size_t d = 0; d < dims.size(); ++d) {
      if (d == dim) {
        if (keepdim) result_indices.push_back(0);
      } else {
        result_indices.push_back(indices[d]);
      }
    }
    if (result_indices.empty()) result_indices.push_back(0);

    size_t result_flat = 0;
    size_t result_stride = 1;
    for (int d = static_cast<int>(result_indices.size()) - 1; d >= 0; --d) {
      result_flat += result_indices[d] * result_stride;
      result_stride *= new_dims[d];
    }

    T v = at(indices);
    if (v > r_data[result_flat]) {
      r_data[result_flat] = v;
    }

    for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d) {
      if (++indices[d] < dims[d]) break;
      indices[d] = 0;
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
  T* r_data = result.data();
  for (size_t i = 0; i < result.numel(); ++i) {
    r_data[i] = std::numeric_limits<T>::max();
  }

  std::vector<size_t> indices(dims.size(), 0);
  for (size_t i = 0; i < numel(); ++i) {
    std::vector<size_t> result_indices;
    for (size_t d = 0; d < dims.size(); ++d) {
      if (d == dim) {
        if (keepdim) result_indices.push_back(0);
      } else {
        result_indices.push_back(indices[d]);
      }
    }
    if (result_indices.empty()) result_indices.push_back(0);

    size_t result_flat = 0;
    size_t result_stride = 1;
    for (int d = static_cast<int>(result_indices.size()) - 1; d >= 0; --d) {
      result_flat += result_indices[d] * result_stride;
      result_stride *= new_dims[d];
    }

    T v = at(indices);
    if (v < r_data[result_flat]) {
      r_data[result_flat] = v;
    }

    for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d) {
      if (++indices[d] < dims[d]) break;
      indices[d] = 0;
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
  for (size_t i = 0; i < numel(); ++i) {
    std::vector<size_t> result_indices;
    for (size_t d = 0; d < dims.size(); ++d) {
      if (d != dim) result_indices.push_back(indices[d]);
    }
    if (result_indices.empty()) result_indices.push_back(0);

    size_t result_flat = 0;
    size_t result_stride = 1;
    for (int d = static_cast<int>(result_indices.size()) - 1; d >= 0; --d) {
      result_flat += result_indices[d] * result_stride;
      result_stride *= new_dims[d];
    }

    T v = at(indices);
    if (v > max_vals[result_flat]) {
      max_vals[result_flat] = v;
      result.data()[result_flat] = indices[dim];
    }

    for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d) {
      if (++indices[d] < dims[d]) break;
      indices[d] = 0;
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
  for (size_t i = 0; i < numel(); ++i) {
    std::vector<size_t> result_indices;
    for (size_t d = 0; d < dims.size(); ++d) {
      if (d != dim) result_indices.push_back(indices[d]);
    }
    if (result_indices.empty()) result_indices.push_back(0);

    size_t result_flat = 0;
    size_t result_stride = 1;
    for (int d = static_cast<int>(result_indices.size()) - 1; d >= 0; --d) {
      result_flat += result_indices[d] * result_stride;
      result_stride *= new_dims[d];
    }

    T v = at(indices);
    if (v < min_vals[result_flat]) {
      min_vals[result_flat] = v;
      result.data()[result_flat] = indices[dim];
    }

    for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d) {
      if (++indices[d] < dims[d]) break;
      indices[d] = 0;
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

#pragma omp parallel for collapse(2) schedule(static)
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

#pragma omp parallel for collapse(3) schedule(static)
    for (size_t b = 0; b < batch_out; ++b) {
      for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
          size_t b1 = (batch1 == 1) ? 0 : b;
          size_t b2 = (batch2 == 1) ? 0 : b;
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

#pragma omp parallel for collapse(4) schedule(static)
    for (size_t b = 0; b < batch_out; ++b) {
      for (size_t h = 0; h < heads_out; ++h) {
        for (size_t i = 0; i < m; ++i) {
          for (size_t j = 0; j < n; ++j) {
            // Compute indices inside loop for OpenMP collapse compatibility
            size_t b1 = (batch1 == 1) ? 0 : b;
            size_t b2 = (batch2 == 1) ? 0 : b;

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
  T* r_data = result.data();

  // Compute exp(x - max)
  std::vector<size_t> indices(dims.size(), 0);
  for (size_t flat_idx = 0; flat_idx < numel(); ++flat_idx) {
    // Get max value for this position
    std::vector<size_t> max_indices;
    for (size_t d = 0; d < dims.size(); ++d) {
      if (d == dim) {
        max_indices.push_back(0);
      } else {
        max_indices.push_back(indices[d]);
      }
    }
    T max_val = max_vals.at(max_indices);
    r_data[flat_idx] = std::exp(at(indices) - max_val);

    for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d) {
      if (++indices[d] < dims[d]) break;
      indices[d] = 0;
    }
  }

  // Compute sum along dimension and divide
  Tensor<T> sum_vals = result.sum(dim, true);

  std::fill(indices.begin(), indices.end(), 0);
  for (size_t flat_idx = 0; flat_idx < numel(); ++flat_idx) {
    std::vector<size_t> sum_indices;
    for (size_t d = 0; d < dims.size(); ++d) {
      if (d == dim) {
        sum_indices.push_back(0);
      } else {
        sum_indices.push_back(indices[d]);
      }
    }
    r_data[flat_idx] /= sum_vals.at(sum_indices);

    for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d) {
      if (++indices[d] < dims[d]) break;
      indices[d] = 0;
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
  T* r_data = result.data();

  // Compute mean along last dimension
  Tensor<T> mean_vals = mean(norm_dim, true);

  // Compute variance: E[(x - mean)^2]
  // We need to compute this without broadcasting, so do it element by element
  Tensor<T> var_vals = Tensor<T>::zeros(mean_vals.shape());

  std::vector<size_t> indices(ndim(), 0);
  for (size_t i = 0; i < numel(); ++i) {
    std::vector<size_t> mean_indices;
    for (size_t d = 0; d < ndim(); ++d) {
      if (d == norm_dim) {
        mean_indices.push_back(0);
      } else {
        mean_indices.push_back(indices[d]);
      }
    }

    T mean_val = mean_vals.at(mean_indices);
    T diff = at(indices) - mean_val;
    var_vals.at(mean_indices) += diff * diff;

    for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
      if (++indices[d] < shape_[d]) break;
      indices[d] = 0;
    }
  }

  // Divide by norm_size to get variance
  T* var_data = var_vals.data();
  for (size_t i = 0; i < var_vals.numel(); ++i) {
    var_data[i] /= static_cast<T>(norm_size);
  }

  // Normalize: (x - mean) / sqrt(var + eps) * weight + bias
  std::fill(indices.begin(), indices.end(), 0);
  for (size_t flat_idx = 0; flat_idx < numel(); ++flat_idx) {
    std::vector<size_t> stat_indices;
    for (size_t d = 0; d < ndim(); ++d) {
      if (d == norm_dim) {
        stat_indices.push_back(0);
      } else {
        stat_indices.push_back(indices[d]);
      }
    }

    T var_val = var_vals.at(stat_indices);
    T mean_val = mean_vals.at(stat_indices);
    size_t weight_idx = indices[norm_dim];

    r_data[flat_idx] = (at(indices) - mean_val) /
                       std::sqrt(var_val + eps) *
                       weight.data()[weight_idx] +
                       bias.data()[weight_idx];

    for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
      if (++indices[d] < shape_[d]) break;
      indices[d] = 0;
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
  (void)norm_size;  // Suppress unused warning

  if (weight.numel() != shape_[norm_dim]) {
    throw std::runtime_error("Weight size must match last dimension");
  }

  Tensor<T> result(shape_);
  T* r_data = result.data();

  // Compute mean of squares along last dimension
  Tensor<T> sq = (*this) * (*this);
  Tensor<T> mean_sq = sq.mean(norm_dim, true);

  // Normalize
  std::vector<size_t> indices(ndim(), 0);
  for (size_t flat_idx = 0; flat_idx < numel(); ++flat_idx) {
    std::vector<size_t> rms_indices;
    for (size_t d = 0; d < ndim(); ++d) {
      if (d == norm_dim) {
        rms_indices.push_back(0);
      } else {
        rms_indices.push_back(indices[d]);
      }
    }

    T mean_sq_val = mean_sq.at(rms_indices);
    T rms = std::sqrt(mean_sq_val + eps);
    size_t weight_idx = indices[norm_dim];

    r_data[flat_idx] = at(indices) / rms * weight.data()[weight_idx];

    for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
      if (++indices[d] < shape_[d]) break;
      indices[d] = 0;
    }
  }

  return result;
}

// Utility functions
template <typename T>
void Tensor<T>::fill(const T& value) {
  const auto& dims = shape_.dims();
  std::vector<size_t> indices(ndim(), 0);
  for (size_t i = 0; i < numel(); ++i) {
    at(indices) = value;
    for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
      if (++indices[d] < dims[d]) break;
      indices[d] = 0;
    }
  }
}

template <typename T>
Tensor<T> Tensor<T>::clone() const {
  Tensor<T> result(shape_);
  T* r_data = result.data();
  if (is_contiguous()) {
    const T* src = data();
    std::copy(src, src + numel(), r_data);
  } else {
    const auto& dims = shape_.dims();
    std::vector<size_t> indices(ndim(), 0);
    for (size_t i = 0; i < numel(); ++i) {
      r_data[i] = at(indices);
      for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
        if (++indices[d] < dims[d]) break;
        indices[d] = 0;
      }
    }
  }
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
Tensor<T> Tensor<T>::concat(const std::vector<Tensor<T>>& tensors, size_t dim) {
  if (tensors.empty()) {
    throw std::runtime_error("Cannot concatenate empty tensor list");
  }

  const Shape& first_shape = tensors[0].shape();
  size_t ndim = first_shape.ndim();

  if (dim >= ndim) {
    throw std::runtime_error("Concatenation dimension out of range");
  }

  // Validate all tensors have same shape except along concat dimension
  size_t total_concat_size = first_shape[dim];
  for (size_t t = 1; t < tensors.size(); ++t) {
    const Shape& s = tensors[t].shape();
    if (s.ndim() != ndim) {
      throw std::runtime_error(
          "All tensors must have the same number of dimensions");
    }
    for (size_t d = 0; d < ndim; ++d) {
      if (d != dim && s[d] != first_shape[d]) {
        throw std::runtime_error(
            "All tensors must have the same shape except along concat dimension");
      }
    }
    total_concat_size += s[dim];
  }

  // Create result shape
  std::vector<size_t> result_dims = first_shape.dims();
  result_dims[dim] = total_concat_size;
  Tensor<T> result{Shape(result_dims)};
  T* r_data = result.data();

  // Compute strides for the result tensor
  std::vector<size_t> result_strides(ndim);
  size_t stride = 1;
  for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
    result_strides[i] = stride;
    stride *= result_dims[i];
  }

  // Copy data from each tensor
  size_t concat_offset = 0;
  for (size_t t = 0; t < tensors.size(); ++t) {
    const Tensor<T>& src = tensors[t];
    const auto& src_dims = src.shape().dims();
    size_t src_concat_size = src_dims[dim];

    // Iterate over all elements in source tensor using strided access
    std::vector<size_t> indices(ndim, 0);
    for (size_t i = 0; i < src.numel(); ++i) {
      // Compute destination index (offset the concat dimension)
      size_t dst_flat = 0;
      for (size_t d = 0; d < ndim; ++d) {
        size_t idx = indices[d];
        if (d == dim) {
          idx += concat_offset;
        }
        dst_flat += idx * result_strides[d];
      }

      r_data[dst_flat] = src.at(indices);

      // Increment indices
      for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
        if (++indices[d] < src_dims[d]) break;
        indices[d] = 0;
      }
    }

    concat_offset += src_concat_size;
  }

  return result;
}

template <typename T>
Tensor<T> Tensor<T>::apply(std::function<T(T)> func) const {
  Tensor<T> result(shape_);
  T* r_data = result.data();
  if (is_contiguous()) {
    const T* src = data();
    for (size_t i = 0; i < numel(); ++i) {
      r_data[i] = func(src[i]);
    }
  } else {
    const auto& dims = shape_.dims();
    std::vector<size_t> indices(ndim(), 0);
    for (size_t i = 0; i < numel(); ++i) {
      r_data[i] = func(at(indices));
      for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
        if (++indices[d] < dims[d]) break;
        indices[d] = 0;
      }
    }
  }
  return result;
}

// RoPE (Rotary Position Embedding) implementation
// Applies rotation to pairs of elements based on position
template <typename T>
Tensor<T> Tensor<T>::rope(size_t start_pos, T base) const {
  // Tensor should have at least 2 dimensions: [..., seq_len, head_dim]
  if (ndim() < 2) {
    throw std::runtime_error("RoPE requires at least 2D tensor");
  }

  // RoPE requires contiguous layout for efficient access
  if (!is_contiguous()) {
    return contiguous().rope(start_pos, base);
  }

  size_t head_dim = shape_[ndim() - 1];
  size_t seq_len = shape_[ndim() - 2];

  if (head_dim % 2 != 0) {
    throw std::runtime_error("RoPE requires even head_dim");
  }

  Tensor<T> result(shape_);
  const T* src_data = data();
  T* r_data = result.data();

  // Compute inverse frequencies: 1 / (base^(2i/d)) for i in [0, head_dim/2)
  std::vector<T> inv_freq(head_dim / 2);
  for (size_t i = 0; i < head_dim / 2; ++i) {
    T exp = static_cast<T>(2 * i) / static_cast<T>(head_dim);
    inv_freq[i] = T(1) / std::pow(base, exp);
  }

  // Number of "instances" (batch * heads or whatever comes before seq_len)
  size_t outer_size = numel() / (seq_len * head_dim);

  // Apply rotation to each position
  for (size_t outer = 0; outer < outer_size; ++outer) {
    size_t outer_offset = outer * seq_len * head_dim;

    for (size_t pos = 0; pos < seq_len; ++pos) {
      size_t abs_pos = start_pos + pos;
      size_t pos_offset = outer_offset + pos * head_dim;

      // Apply rotation to pairs of elements
      for (size_t i = 0; i < head_dim / 2; ++i) {
        T theta = static_cast<T>(abs_pos) * inv_freq[i];
        T cos_theta = std::cos(theta);
        T sin_theta = std::sin(theta);

        // Get the pair of values
        T x0 = src_data[pos_offset + i];
        T x1 = src_data[pos_offset + i + head_dim / 2];

        // Apply rotation: [cos, -sin; sin, cos] @ [x0, x1]
        r_data[pos_offset + i] = x0 * cos_theta - x1 * sin_theta;
        r_data[pos_offset + i + head_dim / 2] = x0 * sin_theta + x1 * cos_theta;
      }
    }
  }

  return result;
}

// Causal mask: mask[i,j] = 0 if j <= i + offset, else -inf
// offset = total_len - seq_len
template <typename T>
Tensor<T> Tensor<T>::causal_mask(size_t seq_len, size_t total_len) {
  Tensor<T> mask(Shape({seq_len, total_len}));
  size_t offset = total_len - seq_len;

  for (size_t i = 0; i < seq_len; ++i) {
    for (size_t j = 0; j < total_len; ++j) {
      // Allow attention to positions j where j <= i + offset
      // i.e., j <= i + (total_len - seq_len)
      // which means j - (total_len - seq_len) <= i
      if (j <= i + offset) {
        mask(i, j) = T(0);
      } else {
        mask(i, j) = -std::numeric_limits<T>::infinity();
      }
    }
  }

  return mask;
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

// Stream output operator
template <typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor) {
  const auto& dims = tensor.shape().dims();
  size_t ndim = dims.size();

  // Print shape
  os << "Tensor(shape=[";
  for (size_t i = 0; i < ndim; ++i) {
    os << dims[i];
    if (i < ndim - 1) os << ", ";
  }
  os << "], data=";

  if (tensor.numel() == 0) {
    os << "[])";
    return os;
  }

  // For large dimensions, only show first and last few elements
  constexpr size_t max_display = 8;

  // Helper to recursively print tensor structure
  std::function<void(size_t, std::vector<size_t>&)> print_recursive;
  print_recursive = [&](size_t dim, std::vector<size_t>& indices) {
    os << "[";
    size_t size = dims[dim];
    bool dim_truncate = size > max_display;

    for (size_t i = 0; i < size; ++i) {
      // Truncate middle elements for large dimensions
      if (dim_truncate && i == max_display / 2 && size > max_display) {
        os << "...";
        i = size - max_display / 2 - 1;
        if (i + 1 < size) os << ", ";
        continue;
      }

      indices[dim] = i;
      if (dim == ndim - 1) {
        // Leaf level - print the value
        os << std::setprecision(4) << tensor.at(indices);
      } else {
        // Recurse to next dimension
        print_recursive(dim + 1, indices);
      }
      if (i < size - 1) os << ", ";
    }
    os << "]";
  };

  std::vector<size_t> indices(ndim, 0);
  print_recursive(0, indices);
  os << ")";

  return os;
}
