# Migrating from Copy-Based Tensors to Stride-Based Views

This document explains how to migrate a tensor library from a "copy everything" architecture to a "shared memory views" architecture, similar to how NumPy and PyTorch work internally.

## Table of Contents

1. [Overview](#overview)
2. [The Problem with Copying](#the-problem-with-copying)
3. [Understanding Strides](#understanding-strides)
4. [Key C++ Concepts](#key-c-concepts)
5. [Migration Steps](#migration-steps)
6. [Code Changes in Detail](#code-changes-in-detail)
7. [Gotchas and Pitfalls](#gotchas-and-pitfalls)
8. [Performance Results](#performance-results)

---

## Overview

### Before: Copy-Based Architecture

```cpp
template <typename T>
class Tensor {
 private:
  Shape shape_;
  std::vector<T> data_;  // Each tensor owns its own data
};
```

Every operation creates a new tensor with a full copy of the data:

```cpp
Tensor<T> reshape(const Shape& new_shape) const {
  Tensor<T> result(new_shape);
  std::copy(data_.begin(), data_.end(), result.data_.begin());  // O(n) copy!
  return result;
}
```

### After: Stride-Based Views Architecture

```cpp
template <typename T>
class Tensor {
 private:
  Shape shape_;
  std::shared_ptr<std::vector<T>> storage_;  // Shared underlying data
  size_t offset_;                             // Start position in storage
  std::vector<size_t> strides_;               // How to navigate the data
};
```

Operations can now return "views" that share the same underlying data:

```cpp
Tensor<T> reshape(const Shape& new_shape) const {
  // O(1) - just create new metadata, share the storage!
  return Tensor<T>(storage_, offset_, new_shape,
                   compute_contiguous_strides(new_shape));
}
```

---

## The Problem with Copying

In a neural network forward pass, you might have code like:

```cpp
// In attention mechanism
Tensor<float> q = input.matmul(q_proj);           // New tensor
Tensor<float> q_reshaped = q.reshape({batch, seq, heads, dim});  // COPY!
Tensor<float> q_transposed = q_reshaped.transpose(1, 2);         // COPY!
// Similar for K, V...
```

For a model like Qwen3-4B with 36 layers, a single forward pass creates:
- 108 reshape copies (Q, K, V × 36 layers)
- 108 transpose copies
- Hundreds more for other operations

Each copy is O(n) where n can be millions of elements. This dominated our runtime.

---

## Understanding Strides

### What Are Strides?

Strides tell you how many elements to skip in memory to move one position along each dimension.

For a contiguous 2D tensor with shape `[3, 4]`:
```
Memory layout: [a, b, c, d, e, f, g, h, i, j, k, l]
               row 0       row 1       row 2

Shape:   [3, 4]
Strides: [4, 1]  // Skip 4 elements to go to next row, 1 to go to next column
```

To access element `[i, j]`: `data[i * strides[0] + j * strides[1]]`
- Element `[0, 0]`: `data[0*4 + 0*1] = data[0]` = a
- Element `[1, 2]`: `data[1*4 + 2*1] = data[6]` = g
- Element `[2, 3]`: `data[2*4 + 3*1] = data[11]` = l

### How Transpose Works with Strides

Instead of copying data, just swap the strides!

```
Original:
  Shape:   [3, 4]
  Strides: [4, 1]

Transposed (swap shape and strides):
  Shape:   [4, 3]
  Strides: [1, 4]  // Now skip 1 to go to next "row", 4 for next "column"
```

Same memory, different interpretation:
- Original `[1, 2]`: `data[1*4 + 2*1] = data[6]`
- Transposed `[2, 1]`: `data[2*1 + 1*4] = data[6]` ✓ Same element!

### Non-Contiguous Memory

After transpose, the data is no longer contiguous in memory:

```
Contiguous [3, 4]:     Reading order: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
Transposed [4, 3]:     Reading order: 0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11
                       (jumping around in memory!)
```

This is why we need `is_contiguous()` and `contiguous()` methods.

---

## Key C++ Concepts

### 1. `std::shared_ptr` for Shared Ownership

```cpp
std::shared_ptr<std::vector<T>> storage_;
```

- Multiple tensors can point to the same underlying data
- Reference counting handles cleanup automatically
- When the last tensor is destroyed, the data is freed

```cpp
Tensor<float> a({3, 4});           // Creates storage, ref_count = 1
Tensor<float> b = a.transpose();   // Shares storage, ref_count = 2
// When 'a' goes out of scope: ref_count = 1
// When 'b' goes out of scope: ref_count = 0, storage freed
```

### 2. Private Constructor for Views

```cpp
private:
  // Only the Tensor class can create views
  Tensor(std::shared_ptr<std::vector<T>> storage, size_t offset,
         const Shape& shape, std::vector<size_t> strides);
```

This constructor is private because:
- External code shouldn't create arbitrary views
- Ensures views are only created by Tensor methods
- Maintains invariants (valid offset, strides, etc.)

### 3. Computing Contiguous Strides

```cpp
static std::vector<size_t> compute_contiguous_strides(const Shape& shape) {
  const auto& dims = shape.dims();
  std::vector<size_t> strides(dims.size());
  size_t stride = 1;
  // Work backwards from last dimension
  for (int i = dims.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= dims[i];
  }
  return strides;
}
```

Example for shape `[2, 3, 4]`:
- Start with stride = 1
- Dimension 2 (size 4): strides[2] = 1, stride = 1 × 4 = 4
- Dimension 1 (size 3): strides[1] = 4, stride = 4 × 3 = 12
- Dimension 0 (size 2): strides[0] = 12, stride = 12 × 2 = 24
- Result: strides = [12, 4, 1]

### 4. Checking Contiguity

```cpp
bool is_contiguous() const {
  size_t expected = 1;
  for (int i = ndim() - 1; i >= 0; --i) {
    if (strides_[i] != expected) return false;
    expected *= shape_[i];
  }
  return true;
}
```

A tensor is contiguous if its strides match what `compute_contiguous_strides` would produce.

---

## Migration Steps

### Step 1: Add New Private Members

```cpp
// Before
private:
  Shape shape_;
  std::vector<T> data_;

// After
private:
  Shape shape_;
  std::shared_ptr<std::vector<T>> storage_;
  size_t offset_ = 0;
  std::vector<size_t> strides_;
```

### Step 2: Add Private View Constructor

```cpp
private:
  Tensor(std::shared_ptr<std::vector<T>> storage, size_t offset,
         const Shape& shape, std::vector<size_t> strides)
      : shape_(shape)
      , storage_(storage)
      , offset_(offset)
      , strides_(std::move(strides)) {}
```

### Step 3: Update Public Constructors

```cpp
// Owning constructor - creates new storage
Tensor(const Shape& s)
    : shape_(s)
    , storage_(std::make_shared<std::vector<T>>(s.numel()))
    , offset_(0)
    , strides_(compute_contiguous_strides(s)) {}

// Fill constructor
Tensor(const Shape& s, const T& fill_value)
    : shape_(s)
    , storage_(std::make_shared<std::vector<T>>(s.numel(), fill_value))
    , offset_(0)
    , strides_(compute_contiguous_strides(s)) {}
```

### Step 4: Update Element Access

```cpp
// Before
T& operator()(size_t i, size_t j) {
  return data_[i * shape_[1] + j];
}

// After - use strides for proper indexing
T& operator()(size_t i, size_t j) {
  return (*storage_)[offset_ + i * strides_[0] + j * strides_[1]];
}
```

### Step 5: Update data() Accessor

```cpp
// Before
T* data() { return data_.data(); }

// After - account for offset
T* data() { return storage_->data() + offset_; }
```

### Step 6: Make Shape Operations Return Views

```cpp
// reshape() - O(1) view instead of O(n) copy
Tensor<T> reshape(const Shape& new_shape) const {
  if (new_shape.numel() != numel()) {
    throw std::runtime_error("Cannot reshape: element count must match");
  }
  if (!is_contiguous()) {
    return contiguous().reshape(new_shape);  // Must copy first
  }
  return Tensor<T>(storage_, offset_, new_shape,
                   compute_contiguous_strides(new_shape));
}

// transpose() - O(1) view, just swap strides
Tensor<T> transpose() const {
  if (ndim() != 2) throw std::runtime_error("Need 2D tensor");
  Shape new_shape({shape_[1], shape_[0]});
  std::vector<size_t> new_strides = {strides_[1], strides_[0]};
  return Tensor<T>(storage_, offset_, new_shape, new_strides);
}

// permute() - O(1) view, reorder strides
Tensor<T> permute(const std::vector<size_t>& order) const {
  std::vector<size_t> new_dims(ndim());
  std::vector<size_t> new_strides(ndim());
  for (size_t i = 0; i < ndim(); ++i) {
    new_dims[i] = shape_[order[i]];
    new_strides[i] = strides_[order[i]];
  }
  return Tensor<T>(storage_, offset_, Shape(new_dims), new_strides);
}

// slice() - O(1) view, adjust offset
Tensor<T> slice(size_t dim, size_t start, size_t end) const {
  std::vector<size_t> new_dims = shape_.dims();
  new_dims[dim] = end - start;
  size_t new_offset = offset_ + start * strides_[dim];
  return Tensor<T>(storage_, new_offset, Shape(new_dims), strides_);
}
```

### Step 7: Add contiguous() Method

```cpp
Tensor<T> contiguous() const {
  if (is_contiguous()) {
    return *this;  // Already contiguous, return view (shallow copy)
  }

  // Create new contiguous storage
  Tensor<T> result(shape_);

  // Copy with stride-aware iteration
  const auto& dims = shape_.dims();
  std::vector<size_t> indices(ndim(), 0);
  for (size_t flat_idx = 0; flat_idx < numel(); ++flat_idx) {
    (*result.storage_)[flat_idx] = at(indices);

    // Increment multi-dimensional index
    for (int i = ndim() - 1; i >= 0; --i) {
      if (++indices[i] < dims[i]) break;
      indices[i] = 0;
    }
  }
  return result;
}
```

### Step 8: Update All Operations to Handle Non-Contiguous Data

For every operation that iterates over data, you have two choices:

**Option A: Require contiguous input**
```cpp
Tensor<T> some_operation() const {
  if (!is_contiguous()) {
    return contiguous().some_operation();
  }
  // ... work with data() directly
}
```

**Option B: Use stride-aware access**
```cpp
Tensor<T> some_operation() const {
  Tensor<T> result(shape_);
  const auto& dims = shape_.dims();
  std::vector<size_t> indices(ndim(), 0);

  for (size_t i = 0; i < numel(); ++i) {
    result.data()[i] = do_something(at(indices));

    // Increment indices
    for (int d = ndim() - 1; d >= 0; --d) {
      if (++indices[d] < dims[d]) break;
      indices[d] = 0;
    }
  }
  return result;
}
```

**Option C: Fast path for contiguous, slow path otherwise**
```cpp
Tensor<T> some_operation() const {
  Tensor<T> result(shape_);
  T* r_data = result.data();

  if (is_contiguous()) {
    const T* src = data();
    for (size_t i = 0; i < numel(); ++i) {
      r_data[i] = do_something(src[i]);
    }
  } else {
    const auto& dims = shape_.dims();
    std::vector<size_t> indices(ndim(), 0);
    for (size_t i = 0; i < numel(); ++i) {
      r_data[i] = do_something(at(indices));
      for (int d = ndim() - 1; d >= 0; --d) {
        if (++indices[d] < dims[d]) break;
        indices[d] = 0;
      }
    }
  }
  return result;
}
```

---

## Code Changes in Detail

### Before and After: Full Example

**Before (copy-based transpose):**
```cpp
Tensor<T> transpose() const {
  if (ndim() != 2) {
    throw std::runtime_error("transpose() requires 2D tensor");
  }
  size_t rows = shape_[0];
  size_t cols = shape_[1];

  // Create new tensor with transposed shape
  Tensor<T> result(Shape({cols, rows}));

  // Copy every element - O(n)!
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      result(j, i) = (*this)(i, j);
    }
  }
  return result;
}
```

**After (stride-based view):**
```cpp
Tensor<T> transpose() const {
  if (ndim() != 2) {
    throw std::runtime_error("transpose() requires 2D tensor");
  }

  // Just swap shape and strides - O(1)!
  Shape new_shape({shape_[1], shape_[0]});
  std::vector<size_t> new_strides = {strides_[1], strides_[0]};

  return Tensor<T>(storage_, offset_, new_shape, new_strides);
}
```

### Multi-Index Iteration Pattern

This pattern appears throughout the codebase for stride-aware iteration:

```cpp
const auto& dims = shape_.dims();
std::vector<size_t> indices(ndim(), 0);

for (size_t flat_idx = 0; flat_idx < numel(); ++flat_idx) {
  // Access element at current indices
  T value = at(indices);

  // ... do something with value ...

  // Increment indices (like an odometer)
  for (int d = ndim() - 1; d >= 0; --d) {
    if (++indices[d] < dims[d]) break;  // No overflow, done
    indices[d] = 0;                      // Overflow, carry to next dimension
  }
}
```

This iterates through all index combinations:
- For shape [2, 3]: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)

---

## Gotchas and Pitfalls

### 1. Flat Indexing with operator()(size_t i)

**Problem:** Tests often use `t(i)` to iterate over all elements:
```cpp
for (size_t i = 0; i < t.numel(); ++i) {
  EXPECT_EQ(t(i), expected[i]);
}
```

**Issue:** If `operator()(size_t i)` uses strides, it breaks for 2D+ tensors:
```cpp
// BAD: Uses strides_[0] which is wrong for 2D tensor
T& operator()(size_t i) {
  return (*storage_)[offset_ + i * strides_[0]];
}

// For shape [2, 3], strides = [3, 1]
// t(0) = storage[0*3] = storage[0] ✓
// t(1) = storage[1*3] = storage[3] ✗ Should be storage[1]!
```

**Solution:** Make flat indexing use `data()[i]` (assumes contiguous):
```cpp
T& operator()(size_t i) {
  return data()[i];  // Flat indexing, assumes contiguous
}
```

### 2. Forgetting to Update Copy Constructor

**Problem:** Default copy creates shallow copy (shared storage):
```cpp
Tensor<T>::Tensor(const Tensor& other)
    : shape_(other.shape_)
    , storage_(other.storage_)   // Shares storage!
    , offset_(other.offset_)
    , strides_(other.strides_) {}
```

**This is actually correct** for views - modifying one affects the other. But be aware:
```cpp
Tensor<float> a({2, 2}, 1.0f);
Tensor<float> b = a;  // b shares storage with a
b(0, 0) = 999.0f;     // This also modifies a!
```

If you need a deep copy, use `clone()`:
```cpp
Tensor<float> b = a.clone();  // Independent copy
```

### 3. Non-Contiguous Data in Operations

**Problem:** Some operations assume contiguous memory:
```cpp
// RoPE uses flat indexing internally
T x0 = data_[pos_offset + i];  // Assumes contiguous!
```

**Solution:** Call `contiguous()` first for such operations:
```cpp
Tensor<T> rope(size_t start_pos, T base) const {
  if (!is_contiguous()) {
    return contiguous().rope(start_pos, base);  // Force contiguous first
  }
  // ... rest of implementation using data() directly
}
```

### 4. Reshape of Non-Contiguous Tensor

**Problem:** Can't create a contiguous view of non-contiguous data:
```cpp
Tensor<float> a({3, 4});
Tensor<float> b = a.transpose();  // Non-contiguous
Tensor<float> c = b.reshape({12}); // Can't view as [12] - data not contiguous!
```

**Solution:** Force contiguous copy:
```cpp
Tensor<T> reshape(const Shape& new_shape) const {
  if (!is_contiguous()) {
    return contiguous().reshape(new_shape);  // Copy then reshape
  }
  // ... return view
}
```

### 5. In-Place Operations on Views

**Problem:** In-place operations might affect shared data:
```cpp
Tensor<float> original({3, 4});
Tensor<float> view = original.slice(0, 0, 2);  // View of first 2 rows
view += 1.0f;  // Modifies original's data too!
```

This is usually the desired behavior (like NumPy), but can surprise developers.

### 6. Reference Counting and Lifetime

**Problem:** View outlives original:
```cpp
Tensor<float>* make_view() {
  Tensor<float> local({3, 4});
  return new Tensor<float>(local.transpose());  // Returns view
}  // local destroyed here, but storage lives on via shared_ptr
```

Thanks to `shared_ptr`, this actually works correctly - the storage stays alive as long as any view references it.

---

## Performance Results

### Benchmark: Qwen3-4B Forward Pass

| Metric | Copy-Based | Stride-Based | Improvement |
|--------|-----------|--------------|-------------|
| Time per token | ~18s | ~2.7s | **6.7x faster** |
| Tokens/second | 0.05 | 0.38 | **7.5x** |
| Memory copies per forward | ~600+ | ~50 | **12x fewer** |

### Why Such a Big Improvement?

1. **reshape()**: O(n) → O(1)
   - Called 3× per layer (Q, K, V) × 36 layers = 108 times
   - Each was copying millions of floats

2. **transpose()**: O(n) → O(1)
   - Called 3× per layer × 36 layers = 108 times
   - Same massive savings

3. **slice()**: O(n) → O(1)
   - Used for KV cache updates
   - Now just adjusts offset pointer

4. **Memory allocation overhead**: Eliminated
   - No more allocating/deallocating temporary buffers

---

## Summary

The key insights for this migration:

1. **Strides decouple logical layout from physical layout** - same data, different interpretation

2. **`shared_ptr` enables safe memory sharing** - automatic cleanup, no dangling pointers

3. **Private view constructor maintains invariants** - only Tensor methods can create views

4. **`contiguous()` is the escape hatch** - when you truly need contiguous data, copy it

5. **Fast path + slow path** - check `is_contiguous()` for optimized hot path

6. **Multi-index iteration** - the odometer pattern works for any number of dimensions

This pattern is used in production tensor libraries like PyTorch, NumPy, and TensorFlow, making it a valuable skill for anyone working with numerical computing in C++.
