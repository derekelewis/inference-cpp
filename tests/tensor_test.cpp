#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "tensor.h"

// Helper for floating point comparison
constexpr float kEpsilon = 1e-5f;

// ============================================================================
// Shape Tests
// ============================================================================

TEST(ShapeTest, ConstructorWithInitializerList) {
  Shape s({2, 3, 4});
  EXPECT_EQ(s.ndim(), 3u);
  EXPECT_EQ(s[0], 2u);
  EXPECT_EQ(s[1], 3u);
  EXPECT_EQ(s[2], 4u);
}

TEST(ShapeTest, ConstructorWithVector) {
  std::vector<size_t> dims = {5, 10};
  Shape s(dims);
  EXPECT_EQ(s.ndim(), 2u);
  EXPECT_EQ(s[0], 5u);
  EXPECT_EQ(s[1], 10u);
}

TEST(ShapeTest, NumelCalculation) {
  Shape s({2, 3, 4});
  EXPECT_EQ(s.numel(), 24u);
}

TEST(ShapeTest, DimsAccessor) {
  Shape s({3, 4});
  const auto& dims = s.dims();
  EXPECT_EQ(dims.size(), 2u);
  EXPECT_EQ(dims[0], 3u);
  EXPECT_EQ(dims[1], 4u);
}

TEST(ShapeTest, EqualityOperator) {
  Shape s1({2, 3});
  Shape s2({2, 3});
  Shape s3({3, 2});
  EXPECT_TRUE(s1 == s2);
  EXPECT_FALSE(s1 == s3);
  EXPECT_TRUE(s1 != s3);
}

TEST(ShapeTest, InvalidShapeThrows) {
  EXPECT_THROW(Shape({}), std::runtime_error);
}

TEST(ShapeTest, IndexOutOfRangeThrows) {
  Shape s({2, 3});
  EXPECT_THROW(s[5], std::out_of_range);
}

// ============================================================================
// Tensor Constructor Tests
// ============================================================================

TEST(TensorConstructorTest, DefaultConstructor) {
  Tensor<float> t;
  EXPECT_EQ(t.numel(), 1u);
  EXPECT_EQ(t.ndim(), 1u);
}

TEST(TensorConstructorTest, ShapeConstructor) {
  Tensor<float> t(Shape({2, 3}));
  EXPECT_EQ(t.numel(), 6u);
  EXPECT_EQ(t.ndim(), 2u);
  EXPECT_EQ(t.size(0), 2u);
  EXPECT_EQ(t.size(1), 3u);
}

TEST(TensorConstructorTest, FillValueConstructor) {
  Tensor<float> t(Shape({2, 3}), 5.0f);
  EXPECT_EQ(t.numel(), 6u);
  for (size_t i = 0; i < t.numel(); ++i) {
    EXPECT_FLOAT_EQ(t(i), 5.0f);
  }
}

TEST(TensorConstructorTest, DataVectorConstructor) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor<float> t(Shape({2, 3}), data);
  EXPECT_EQ(t.numel(), 6u);
  for (size_t i = 0; i < t.numel(); ++i) {
    EXPECT_FLOAT_EQ(t(i), data[i]);
  }
}

TEST(TensorConstructorTest, CopyConstructor) {
  Tensor<float> t1(Shape({2, 3}), 3.0f);
  Tensor<float> t2(t1);
  EXPECT_EQ(t2.numel(), 6u);
  for (size_t i = 0; i < t2.numel(); ++i) {
    EXPECT_FLOAT_EQ(t2(i), 3.0f);
  }
}

TEST(TensorConstructorTest, MoveConstructor) {
  Tensor<float> t1(Shape({2, 3}), 3.0f);
  Tensor<float> t2(std::move(t1));
  EXPECT_EQ(t2.numel(), 6u);
  for (size_t i = 0; i < t2.numel(); ++i) {
    EXPECT_FLOAT_EQ(t2(i), 3.0f);
  }
}

TEST(TensorConstructorTest, DataSizeMismatchThrows) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  EXPECT_THROW(Tensor<float>(Shape({2, 3}), data), std::runtime_error);
}

// ============================================================================
// Data Access Tests
// ============================================================================

TEST(TensorAccessTest, DataPointer) {
  Tensor<float> t(Shape({2, 3}), 1.0f);
  float* data = t.data();
  EXPECT_NE(data, nullptr);
  data[0] = 5.0f;
  EXPECT_FLOAT_EQ(t(0), 5.0f);
}

TEST(TensorAccessTest, OneDimensionalAccess) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  Tensor<float> t(Shape({3}), data);
  EXPECT_FLOAT_EQ(t(0), 1.0f);
  EXPECT_FLOAT_EQ(t(1), 2.0f);
  EXPECT_FLOAT_EQ(t(2), 3.0f);
}

TEST(TensorAccessTest, TwoDimensionalAccess) {
  // Row-major: [[1, 2, 3], [4, 5, 6]]
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor<float> t(Shape({2, 3}), data);
  EXPECT_FLOAT_EQ(t(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(t(0, 2), 3.0f);
  EXPECT_FLOAT_EQ(t(1, 0), 4.0f);
  EXPECT_FLOAT_EQ(t(1, 2), 6.0f);
}

TEST(TensorAccessTest, ThreeDimensionalAccess) {
  Tensor<float> t(Shape({2, 3, 4}));
  t(1, 2, 3) = 42.0f;
  EXPECT_FLOAT_EQ(t(1, 2, 3), 42.0f);
}

TEST(TensorAccessTest, FourDimensionalAccess) {
  Tensor<float> t(Shape({2, 3, 4, 5}));
  t(1, 2, 3, 4) = 99.0f;
  EXPECT_FLOAT_EQ(t(1, 2, 3, 4), 99.0f);
}

TEST(TensorAccessTest, AtMethodAccess) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor<float> t(Shape({2, 3}), data);
  EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(t.at({1, 2}), 6.0f);
}

TEST(TensorAccessTest, AtMethodWrite) {
  Tensor<float> t(Shape({2, 3}));
  t.at({1, 1}) = 7.0f;
  EXPECT_FLOAT_EQ(t.at({1, 1}), 7.0f);
}

// ============================================================================
// Shape Manipulation Tests
// ============================================================================

TEST(TensorReshapeTest, BasicReshape) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor<float> t(Shape({2, 3}), data);
  Tensor<float> reshaped = t.reshape(Shape({3, 2}));
  EXPECT_EQ(reshaped.size(0), 3u);
  EXPECT_EQ(reshaped.size(1), 2u);
  // Data order preserved
  EXPECT_FLOAT_EQ(reshaped(0), 1.0f);
  EXPECT_FLOAT_EQ(reshaped(5), 6.0f);
}

TEST(TensorReshapeTest, ReshapeToFlat) {
  Tensor<float> t(Shape({2, 3}), 1.0f);
  Tensor<float> flat = t.reshape(Shape({6}));
  EXPECT_EQ(flat.ndim(), 1u);
  EXPECT_EQ(flat.numel(), 6u);
}

TEST(TensorReshapeTest, InvalidReshapeThrows) {
  Tensor<float> t(Shape({2, 3}));
  EXPECT_THROW(t.reshape(Shape({5})), std::runtime_error);
}

TEST(TensorTransposeTest, Basic2DTranspose) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor<float> t(Shape({2, 3}), data);
  Tensor<float> transposed = t.transpose();
  EXPECT_EQ(transposed.size(0), 3u);
  EXPECT_EQ(transposed.size(1), 2u);
  EXPECT_FLOAT_EQ(transposed(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(transposed(0, 1), 4.0f);
  EXPECT_FLOAT_EQ(transposed(2, 0), 3.0f);
  EXPECT_FLOAT_EQ(transposed(2, 1), 6.0f);
}

TEST(TensorTransposeTest, TransposeWithDims) {
  Tensor<float> t(Shape({2, 3, 4}));
  for (size_t i = 0; i < t.numel(); ++i) {
    t(i) = static_cast<float>(i);
  }
  Tensor<float> transposed = t.transpose(0, 2);
  EXPECT_EQ(transposed.size(0), 4u);
  EXPECT_EQ(transposed.size(1), 3u);
  EXPECT_EQ(transposed.size(2), 2u);
}

TEST(TensorPermuteTest, BasicPermute) {
  Tensor<float> t(Shape({2, 3, 4}));
  for (size_t i = 0; i < t.numel(); ++i) {
    t(i) = static_cast<float>(i);
  }
  Tensor<float> permuted = t.permute({2, 0, 1});
  EXPECT_EQ(permuted.size(0), 4u);
  EXPECT_EQ(permuted.size(1), 2u);
  EXPECT_EQ(permuted.size(2), 3u);
}

TEST(TensorSqueezeTest, SqueezeDimension) {
  Tensor<float> t(Shape({2, 1, 3}), 1.0f);
  Tensor<float> squeezed = t.squeeze(1);
  EXPECT_EQ(squeezed.ndim(), 2u);
  EXPECT_EQ(squeezed.size(0), 2u);
  EXPECT_EQ(squeezed.size(1), 3u);
}

TEST(TensorSqueezeTest, SqueezeInvalidDimThrows) {
  Tensor<float> t(Shape({2, 3}));
  EXPECT_THROW(t.squeeze(0), std::runtime_error);
}

TEST(TensorUnsqueezeTest, UnsqueezeDimension) {
  Tensor<float> t(Shape({2, 3}), 1.0f);
  Tensor<float> unsqueezed = t.unsqueeze(1);
  EXPECT_EQ(unsqueezed.ndim(), 3u);
  EXPECT_EQ(unsqueezed.size(0), 2u);
  EXPECT_EQ(unsqueezed.size(1), 1u);
  EXPECT_EQ(unsqueezed.size(2), 3u);
}

TEST(TensorSliceTest, BasicSlice) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor<float> t(Shape({2, 3}), data);
  Tensor<float> sliced = t.slice(1, 0, 2);
  EXPECT_EQ(sliced.size(0), 2u);
  EXPECT_EQ(sliced.size(1), 2u);
  EXPECT_FLOAT_EQ(sliced(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(sliced(0, 1), 2.0f);
  EXPECT_FLOAT_EQ(sliced(1, 0), 4.0f);
  EXPECT_FLOAT_EQ(sliced(1, 1), 5.0f);
}

TEST(TensorSliceTest, InvalidSliceThrows) {
  Tensor<float> t(Shape({2, 3}));
  EXPECT_THROW(t.slice(0, 5, 10), std::runtime_error);
}

TEST(TensorContiguousTest, ContiguousReturnsClone) {
  Tensor<float> t(Shape({2, 3}), 5.0f);
  Tensor<float> c = t.contiguous();
  EXPECT_EQ(c.numel(), t.numel());
  for (size_t i = 0; i < t.numel(); ++i) {
    EXPECT_FLOAT_EQ(c(i), t(i));
  }
}

// ============================================================================
// Element-wise Operations Tests
// ============================================================================

TEST(TensorElementWiseTest, TensorAddition) {
  Tensor<float> a(Shape({2, 3}), 1.0f);
  Tensor<float> b(Shape({2, 3}), 2.0f);
  Tensor<float> c = a + b;
  for (size_t i = 0; i < c.numel(); ++i) {
    EXPECT_FLOAT_EQ(c(i), 3.0f);
  }
}

TEST(TensorElementWiseTest, TensorSubtraction) {
  Tensor<float> a(Shape({2, 3}), 5.0f);
  Tensor<float> b(Shape({2, 3}), 2.0f);
  Tensor<float> c = a - b;
  for (size_t i = 0; i < c.numel(); ++i) {
    EXPECT_FLOAT_EQ(c(i), 3.0f);
  }
}

TEST(TensorElementWiseTest, TensorMultiplication) {
  Tensor<float> a(Shape({2, 3}), 3.0f);
  Tensor<float> b(Shape({2, 3}), 4.0f);
  Tensor<float> c = a * b;
  for (size_t i = 0; i < c.numel(); ++i) {
    EXPECT_FLOAT_EQ(c(i), 12.0f);
  }
}

TEST(TensorElementWiseTest, TensorDivision) {
  Tensor<float> a(Shape({2, 3}), 12.0f);
  Tensor<float> b(Shape({2, 3}), 4.0f);
  Tensor<float> c = a / b;
  for (size_t i = 0; i < c.numel(); ++i) {
    EXPECT_FLOAT_EQ(c(i), 3.0f);
  }
}

TEST(TensorElementWiseTest, ScalarAddition) {
  Tensor<float> a(Shape({2, 3}), 1.0f);
  Tensor<float> b = a + 5.0f;
  for (size_t i = 0; i < b.numel(); ++i) {
    EXPECT_FLOAT_EQ(b(i), 6.0f);
  }
}

TEST(TensorElementWiseTest, ScalarSubtraction) {
  Tensor<float> a(Shape({2, 3}), 10.0f);
  Tensor<float> b = a - 3.0f;
  for (size_t i = 0; i < b.numel(); ++i) {
    EXPECT_FLOAT_EQ(b(i), 7.0f);
  }
}

TEST(TensorElementWiseTest, ScalarMultiplication) {
  Tensor<float> a(Shape({2, 3}), 5.0f);
  Tensor<float> b = a * 2.0f;
  for (size_t i = 0; i < b.numel(); ++i) {
    EXPECT_FLOAT_EQ(b(i), 10.0f);
  }
}

TEST(TensorElementWiseTest, ScalarDivision) {
  Tensor<float> a(Shape({2, 3}), 10.0f);
  Tensor<float> b = a / 2.0f;
  for (size_t i = 0; i < b.numel(); ++i) {
    EXPECT_FLOAT_EQ(b(i), 5.0f);
  }
}

TEST(TensorElementWiseTest, Negation) {
  Tensor<float> a(Shape({2, 3}), 5.0f);
  Tensor<float> b = a.neg();
  for (size_t i = 0; i < b.numel(); ++i) {
    EXPECT_FLOAT_EQ(b(i), -5.0f);
  }
}

TEST(TensorElementWiseTest, MismatchedShapesThrows) {
  Tensor<float> a(Shape({2, 3}));
  Tensor<float> b(Shape({3, 2}));
  EXPECT_THROW(a + b, std::runtime_error);
}

// In-place operations
TEST(TensorInPlaceTest, InPlaceAddition) {
  Tensor<float> a(Shape({2, 3}), 1.0f);
  Tensor<float> b(Shape({2, 3}), 2.0f);
  a += b;
  for (size_t i = 0; i < a.numel(); ++i) {
    EXPECT_FLOAT_EQ(a(i), 3.0f);
  }
}

TEST(TensorInPlaceTest, InPlaceScalarAddition) {
  Tensor<float> a(Shape({2, 3}), 1.0f);
  a += 5.0f;
  for (size_t i = 0; i < a.numel(); ++i) {
    EXPECT_FLOAT_EQ(a(i), 6.0f);
  }
}

TEST(TensorInPlaceTest, InPlaceSubtraction) {
  Tensor<float> a(Shape({2, 3}), 5.0f);
  Tensor<float> b(Shape({2, 3}), 2.0f);
  a -= b;
  for (size_t i = 0; i < a.numel(); ++i) {
    EXPECT_FLOAT_EQ(a(i), 3.0f);
  }
}

TEST(TensorInPlaceTest, InPlaceMultiplication) {
  Tensor<float> a(Shape({2, 3}), 3.0f);
  Tensor<float> b(Shape({2, 3}), 4.0f);
  a *= b;
  for (size_t i = 0; i < a.numel(); ++i) {
    EXPECT_FLOAT_EQ(a(i), 12.0f);
  }
}

TEST(TensorInPlaceTest, InPlaceDivision) {
  Tensor<float> a(Shape({2, 3}), 12.0f);
  Tensor<float> b(Shape({2, 3}), 4.0f);
  a /= b;
  for (size_t i = 0; i < a.numel(); ++i) {
    EXPECT_FLOAT_EQ(a(i), 3.0f);
  }
}

// Free function operators (scalar on left)
TEST(TensorElementWiseTest, ScalarLeftAddition) {
  Tensor<float> a(Shape({2, 3}), 1.0f);
  Tensor<float> b = 5.0f + a;
  for (size_t i = 0; i < b.numel(); ++i) {
    EXPECT_FLOAT_EQ(b(i), 6.0f);
  }
}

TEST(TensorElementWiseTest, ScalarLeftMultiplication) {
  Tensor<float> a(Shape({2, 3}), 3.0f);
  Tensor<float> b = 2.0f * a;
  for (size_t i = 0; i < b.numel(); ++i) {
    EXPECT_FLOAT_EQ(b(i), 6.0f);
  }
}

// ============================================================================
// Activation Function Tests
// ============================================================================

TEST(TensorActivationTest, ReLU) {
  std::vector<float> data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
  Tensor<float> t(Shape({6}), data);
  Tensor<float> result = t.relu();
  EXPECT_FLOAT_EQ(result(0), 0.0f);
  EXPECT_FLOAT_EQ(result(1), 0.0f);
  EXPECT_FLOAT_EQ(result(2), 0.0f);
  EXPECT_FLOAT_EQ(result(3), 1.0f);
  EXPECT_FLOAT_EQ(result(4), 2.0f);
  EXPECT_FLOAT_EQ(result(5), 3.0f);
}

TEST(TensorActivationTest, Sigmoid) {
  Tensor<float> t(Shape({3}));
  t(0) = 0.0f;
  t(1) = 1.0f;
  t(2) = -1.0f;
  Tensor<float> result = t.sigmoid();
  EXPECT_NEAR(result(0), 0.5f, kEpsilon);
  EXPECT_NEAR(result(1), 0.7310586f, kEpsilon);
  EXPECT_NEAR(result(2), 0.2689414f, kEpsilon);
}

TEST(TensorActivationTest, Tanh) {
  Tensor<float> t(Shape({3}));
  t(0) = 0.0f;
  t(1) = 1.0f;
  t(2) = -1.0f;
  Tensor<float> result = t.tanh_();
  EXPECT_NEAR(result(0), 0.0f, kEpsilon);
  EXPECT_NEAR(result(1), std::tanh(1.0f), kEpsilon);
  EXPECT_NEAR(result(2), std::tanh(-1.0f), kEpsilon);
}

TEST(TensorActivationTest, GELU) {
  Tensor<float> t(Shape({3}));
  t(0) = 0.0f;
  t(1) = 1.0f;
  t(2) = -1.0f;
  Tensor<float> result = t.gelu();
  // GELU(0) = 0
  EXPECT_NEAR(result(0), 0.0f, kEpsilon);
  // GELU(1) ≈ 0.8413
  EXPECT_NEAR(result(1), 0.8413f, 0.01f);
  // GELU(-1) ≈ -0.1587
  EXPECT_NEAR(result(2), -0.1587f, 0.01f);
}

TEST(TensorActivationTest, SiLU) {
  Tensor<float> t(Shape({3}));
  t(0) = 0.0f;
  t(1) = 1.0f;
  t(2) = -1.0f;
  Tensor<float> result = t.silu();
  // SiLU(0) = 0
  EXPECT_NEAR(result(0), 0.0f, kEpsilon);
  // SiLU(x) = x * sigmoid(x)
  EXPECT_NEAR(result(1), 1.0f * 0.7310586f, kEpsilon);
  EXPECT_NEAR(result(2), -1.0f * 0.2689414f, kEpsilon);
}

// ============================================================================
// Reduction Tests
// ============================================================================

TEST(TensorReductionTest, SumAll) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor<float> t(Shape({2, 3}), data);
  float sum = t.sum();
  EXPECT_FLOAT_EQ(sum, 21.0f);
}

TEST(TensorReductionTest, MeanAll) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor<float> t(Shape({2, 3}), data);
  float mean = t.mean();
  EXPECT_FLOAT_EQ(mean, 3.5f);
}

TEST(TensorReductionTest, MaxAll) {
  std::vector<float> data = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
  Tensor<float> t(Shape({2, 3}), data);
  float max = t.max();
  EXPECT_FLOAT_EQ(max, 6.0f);
}

TEST(TensorReductionTest, MinAll) {
  std::vector<float> data = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
  Tensor<float> t(Shape({2, 3}), data);
  float min = t.min();
  EXPECT_FLOAT_EQ(min, 1.0f);
}

TEST(TensorReductionTest, SumAlongDim) {
  // [[1, 2, 3], [4, 5, 6]]
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor<float> t(Shape({2, 3}), data);

  // Sum along dim 0: [5, 7, 9]
  Tensor<float> sum0 = t.sum(0);
  EXPECT_EQ(sum0.ndim(), 1u);
  EXPECT_EQ(sum0.size(0), 3u);
  EXPECT_FLOAT_EQ(sum0(0), 5.0f);
  EXPECT_FLOAT_EQ(sum0(1), 7.0f);
  EXPECT_FLOAT_EQ(sum0(2), 9.0f);

  // Sum along dim 1: [6, 15]
  Tensor<float> sum1 = t.sum(1);
  EXPECT_EQ(sum1.ndim(), 1u);
  EXPECT_EQ(sum1.size(0), 2u);
  EXPECT_FLOAT_EQ(sum1(0), 6.0f);
  EXPECT_FLOAT_EQ(sum1(1), 15.0f);
}

TEST(TensorReductionTest, SumAlongDimKeepdim) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor<float> t(Shape({2, 3}), data);

  Tensor<float> sum0 = t.sum(0, true);
  EXPECT_EQ(sum0.ndim(), 2u);
  EXPECT_EQ(sum0.size(0), 1u);
  EXPECT_EQ(sum0.size(1), 3u);
}

TEST(TensorReductionTest, MeanAlongDim) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor<float> t(Shape({2, 3}), data);

  Tensor<float> mean1 = t.mean(1);
  EXPECT_EQ(mean1.size(0), 2u);
  EXPECT_FLOAT_EQ(mean1(0), 2.0f);
  EXPECT_FLOAT_EQ(mean1(1), 5.0f);
}

TEST(TensorReductionTest, MaxAlongDim) {
  std::vector<float> data = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
  Tensor<float> t(Shape({2, 3}), data);

  Tensor<float> max1 = t.max(1);
  EXPECT_EQ(max1.size(0), 2u);
  EXPECT_FLOAT_EQ(max1(0), 5.0f);
  EXPECT_FLOAT_EQ(max1(1), 6.0f);
}

TEST(TensorReductionTest, MinAlongDim) {
  std::vector<float> data = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
  Tensor<float> t(Shape({2, 3}), data);

  Tensor<float> min1 = t.min(1);
  EXPECT_EQ(min1.size(0), 2u);
  EXPECT_FLOAT_EQ(min1(0), 1.0f);
  EXPECT_FLOAT_EQ(min1(1), 2.0f);
}

TEST(TensorReductionTest, ArgMax) {
  std::vector<float> data = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
  Tensor<float> t(Shape({2, 3}), data);

  Tensor<size_t> argmax1 = t.argmax(1);
  EXPECT_EQ(argmax1.size(0), 2u);
  EXPECT_EQ(argmax1(0), 1u);  // max at index 1 in first row
  EXPECT_EQ(argmax1(1), 2u);  // max at index 2 in second row
}

TEST(TensorReductionTest, ArgMin) {
  std::vector<float> data = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
  Tensor<float> t(Shape({2, 3}), data);

  Tensor<size_t> argmin1 = t.argmin(1);
  EXPECT_EQ(argmin1.size(0), 2u);
  EXPECT_EQ(argmin1(0), 0u);  // min at index 0 in first row
  EXPECT_EQ(argmin1(1), 1u);  // min at index 1 in second row
}

// ============================================================================
// Matrix Multiplication Tests
// ============================================================================

TEST(TensorMatmulTest, Basic2DMatmul) {
  // A: 2x3, B: 3x2
  std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> b_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor<float> a(Shape({2, 3}), a_data);
  Tensor<float> b(Shape({3, 2}), b_data);

  Tensor<float> c = a.matmul(b);
  EXPECT_EQ(c.size(0), 2u);
  EXPECT_EQ(c.size(1), 2u);

  // c[0,0] = 1*1 + 2*3 + 3*5 = 22
  EXPECT_FLOAT_EQ(c(0, 0), 22.0f);
  // c[0,1] = 1*2 + 2*4 + 3*6 = 28
  EXPECT_FLOAT_EQ(c(0, 1), 28.0f);
  // c[1,0] = 4*1 + 5*3 + 6*5 = 49
  EXPECT_FLOAT_EQ(c(1, 0), 49.0f);
  // c[1,1] = 4*2 + 5*4 + 6*6 = 64
  EXPECT_FLOAT_EQ(c(1, 1), 64.0f);
}

TEST(TensorMatmulTest, IdentityMatmul) {
  std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> i_data = {1.0f, 0.0f, 0.0f, 1.0f};
  Tensor<float> a(Shape({2, 2}), a_data);
  Tensor<float> identity(Shape({2, 2}), i_data);

  Tensor<float> c = a.matmul(identity);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(c(i), a_data[i]);
  }
}

TEST(TensorMatmulTest, Batched3DMatmul) {
  // Batch of 2 matrices: (2, 2, 3) x (2, 3, 2)
  Tensor<float> a(Shape({2, 2, 3}), 1.0f);
  Tensor<float> b(Shape({2, 3, 2}), 1.0f);

  Tensor<float> c = a.matmul(b);
  EXPECT_EQ(c.size(0), 2u);
  EXPECT_EQ(c.size(1), 2u);
  EXPECT_EQ(c.size(2), 2u);

  // Each element should be 3.0 (sum of 3 ones)
  for (size_t i = 0; i < c.numel(); ++i) {
    EXPECT_FLOAT_EQ(c(i), 3.0f);
  }
}

TEST(TensorMatmulTest, Batched4DMatmul) {
  // (batch, heads, seq, dim) x (batch, heads, dim, k)
  Tensor<float> a(Shape({2, 4, 3, 5}), 1.0f);
  Tensor<float> b(Shape({2, 4, 5, 6}), 1.0f);

  Tensor<float> c = a.matmul(b);
  EXPECT_EQ(c.size(0), 2u);
  EXPECT_EQ(c.size(1), 4u);
  EXPECT_EQ(c.size(2), 3u);
  EXPECT_EQ(c.size(3), 6u);

  // Each element should be 5.0
  for (size_t i = 0; i < c.numel(); ++i) {
    EXPECT_FLOAT_EQ(c(i), 5.0f);
  }
}

TEST(TensorMatmulTest, DimensionMismatchThrows) {
  Tensor<float> a(Shape({2, 3}));
  Tensor<float> b(Shape({4, 2}));
  EXPECT_THROW(a.matmul(b), std::runtime_error);
}

// ============================================================================
// Normalization Tests
// ============================================================================

TEST(TensorNormalizationTest, Softmax) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  Tensor<float> t(Shape({3}), data);
  Tensor<float> result = t.softmax(0);

  // Sum should be 1
  float sum = result.sum();
  EXPECT_NEAR(sum, 1.0f, kEpsilon);

  // Values should be in ascending order
  EXPECT_LT(result(0), result(1));
  EXPECT_LT(result(1), result(2));
}

TEST(TensorNormalizationTest, Softmax2D) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor<float> t(Shape({2, 3}), data);

  // Softmax along dim 1 (each row sums to 1)
  Tensor<float> result = t.softmax(1);

  // Check each row sums to 1
  float row0_sum = result(0, 0) + result(0, 1) + result(0, 2);
  float row1_sum = result(1, 0) + result(1, 1) + result(1, 2);
  EXPECT_NEAR(row0_sum, 1.0f, kEpsilon);
  EXPECT_NEAR(row1_sum, 1.0f, kEpsilon);
}

TEST(TensorNormalizationTest, LayerNorm) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor<float> t(Shape({2, 3}), data);

  Tensor<float> weight(Shape({3}), 1.0f);
  Tensor<float> bias(Shape({3}), 0.0f);

  Tensor<float> result = t.layer_norm(weight, bias);

  // After layer norm with gamma=1, beta=0:
  // Each row should have mean ≈ 0 and std ≈ 1
  float row0_mean = (result(0, 0) + result(0, 1) + result(0, 2)) / 3.0f;
  float row1_mean = (result(1, 0) + result(1, 1) + result(1, 2)) / 3.0f;
  EXPECT_NEAR(row0_mean, 0.0f, 0.01f);
  EXPECT_NEAR(row1_mean, 0.0f, 0.01f);
}

TEST(TensorNormalizationTest, RMSNorm) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor<float> t(Shape({2, 3}), data);

  Tensor<float> weight(Shape({3}), 1.0f);

  Tensor<float> result = t.rms_norm(weight);

  // RMS norm should normalize by sqrt(mean(x^2))
  // Result values should be scaled versions of original
  EXPECT_GT(result(0, 0), 0.0f);  // Same sign as input
  EXPECT_GT(result(1, 2), 0.0f);
}

// ============================================================================
// Utility Function Tests
// ============================================================================

TEST(TensorUtilityTest, Fill) {
  Tensor<float> t(Shape({2, 3}));
  t.fill(7.0f);
  for (size_t i = 0; i < t.numel(); ++i) {
    EXPECT_FLOAT_EQ(t(i), 7.0f);
  }
}

TEST(TensorUtilityTest, Clone) {
  Tensor<float> t(Shape({2, 3}), 5.0f);
  Tensor<float> c = t.clone();

  // Same values
  for (size_t i = 0; i < t.numel(); ++i) {
    EXPECT_FLOAT_EQ(c(i), t(i));
  }

  // Modifying clone doesn't affect original
  c(0) = 100.0f;
  EXPECT_FLOAT_EQ(t(0), 5.0f);
}

TEST(TensorUtilityTest, Zeros) {
  Tensor<float> t = Tensor<float>::zeros(Shape({3, 4}));
  EXPECT_EQ(t.numel(), 12u);
  for (size_t i = 0; i < t.numel(); ++i) {
    EXPECT_FLOAT_EQ(t(i), 0.0f);
  }
}

TEST(TensorUtilityTest, Ones) {
  Tensor<float> t = Tensor<float>::ones(Shape({3, 4}));
  EXPECT_EQ(t.numel(), 12u);
  for (size_t i = 0; i < t.numel(); ++i) {
    EXPECT_FLOAT_EQ(t(i), 1.0f);
  }
}

TEST(TensorUtilityTest, Full) {
  Tensor<float> t = Tensor<float>::full(Shape({2, 3}), 42.0f);
  for (size_t i = 0; i < t.numel(); ++i) {
    EXPECT_FLOAT_EQ(t(i), 42.0f);
  }
}

TEST(TensorUtilityTest, Apply) {
  std::vector<float> data = {1.0f, 4.0f, 9.0f, 16.0f};
  Tensor<float> t(Shape({4}), data);
  Tensor<float> result = t.apply([](float x) { return std::sqrt(x); });

  EXPECT_FLOAT_EQ(result(0), 1.0f);
  EXPECT_FLOAT_EQ(result(1), 2.0f);
  EXPECT_FLOAT_EQ(result(2), 3.0f);
  EXPECT_FLOAT_EQ(result(3), 4.0f);
}

// ============================================================================
// Integer Tensor Tests
// ============================================================================

TEST(TensorIntTest, BasicOperations) {
  Tensor<int> a(Shape({2, 3}), 5);
  Tensor<int> b(Shape({2, 3}), 3);

  Tensor<int> sum = a + b;
  Tensor<int> diff = a - b;
  Tensor<int> prod = a * b;
  Tensor<int> quot = a / b;

  for (size_t i = 0; i < sum.numel(); ++i) {
    EXPECT_EQ(sum(i), 8);
    EXPECT_EQ(diff(i), 2);
    EXPECT_EQ(prod(i), 15);
    EXPECT_EQ(quot(i), 1);  // Integer division
  }
}

TEST(TensorIntTest, Reductions) {
  std::vector<int> data = {1, 2, 3, 4, 5, 6};
  Tensor<int> t(Shape({6}), data);

  EXPECT_EQ(t.sum(), 21);
  EXPECT_EQ(t.max(), 6);
  EXPECT_EQ(t.min(), 1);
}

// ============================================================================
// Double Precision Tests
// ============================================================================

TEST(TensorDoubleTest, BasicOperations) {
  Tensor<double> a(Shape({2, 3}), 1.5);
  Tensor<double> b(Shape({2, 3}), 2.5);

  Tensor<double> sum = a + b;
  for (size_t i = 0; i < sum.numel(); ++i) {
    EXPECT_DOUBLE_EQ(sum(i), 4.0);
  }
}

TEST(TensorDoubleTest, ActivationFunctions) {
  Tensor<double> t(Shape({3}));
  t(0) = 0.0;
  t(1) = 1.0;
  t(2) = -1.0;

  Tensor<double> sig = t.sigmoid();
  EXPECT_NEAR(sig(0), 0.5, 1e-10);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(TensorEdgeCaseTest, SingleElementTensor) {
  Tensor<float> t(Shape({1}), 42.0f);
  EXPECT_FLOAT_EQ(t.sum(), 42.0f);
  EXPECT_FLOAT_EQ(t.mean(), 42.0f);
  EXPECT_FLOAT_EQ(t.max(), 42.0f);
  EXPECT_FLOAT_EQ(t.min(), 42.0f);
}

TEST(TensorEdgeCaseTest, LargeTensor) {
  Tensor<float> t(Shape({100, 100}), 1.0f);
  EXPECT_FLOAT_EQ(t.sum(), 10000.0f);
  EXPECT_FLOAT_EQ(t.mean(), 1.0f);
}

TEST(TensorEdgeCaseTest, HighDimensionalTensor) {
  Tensor<float> t(Shape({2, 3, 4, 5, 6}));
  EXPECT_EQ(t.numel(), 720u);
  EXPECT_EQ(t.ndim(), 5u);
}

// ============================================================================
// Broadcasting Tests
// ============================================================================

TEST(BroadcastingTest, CanBroadcastSameShape) {
  Shape a({2, 3, 4});
  Shape b({2, 3, 4});
  EXPECT_TRUE(Shape::can_broadcast(a, b));
}

TEST(BroadcastingTest, CanBroadcastScalar) {
  Shape a({2, 3, 4});
  Shape b({1});
  EXPECT_TRUE(Shape::can_broadcast(a, b));
}

TEST(BroadcastingTest, CanBroadcastTrailingDims) {
  Shape a({2, 3, 4});
  Shape b({4});
  EXPECT_TRUE(Shape::can_broadcast(a, b));
}

TEST(BroadcastingTest, CanBroadcastMiddleDim) {
  Shape a({2, 3, 4});
  Shape b({1, 3, 1});
  EXPECT_TRUE(Shape::can_broadcast(a, b));
}

TEST(BroadcastingTest, CanBroadcastDifferentNdim) {
  Shape a({3, 4});
  Shape b({2, 3, 4});
  EXPECT_TRUE(Shape::can_broadcast(a, b));
}

TEST(BroadcastingTest, CannotBroadcastIncompatible) {
  Shape a({2, 3});
  Shape b({4, 3});
  EXPECT_FALSE(Shape::can_broadcast(a, b));
}

TEST(BroadcastingTest, BroadcastShapeSame) {
  Shape a({2, 3, 4});
  Shape b({2, 3, 4});
  Shape result = Shape::broadcast_shape(a, b);
  EXPECT_EQ(result[0], 2u);
  EXPECT_EQ(result[1], 3u);
  EXPECT_EQ(result[2], 4u);
}

TEST(BroadcastingTest, BroadcastShapeExpand) {
  Shape a({2, 1, 4});
  Shape b({1, 3, 4});
  Shape result = Shape::broadcast_shape(a, b);
  EXPECT_EQ(result[0], 2u);
  EXPECT_EQ(result[1], 3u);
  EXPECT_EQ(result[2], 4u);
}

TEST(BroadcastingTest, BroadcastShapeDifferentNdim) {
  Shape a({4});
  Shape b({2, 3, 4});
  Shape result = Shape::broadcast_shape(a, b);
  EXPECT_EQ(result.ndim(), 3u);
  EXPECT_EQ(result[0], 2u);
  EXPECT_EQ(result[1], 3u);
  EXPECT_EQ(result[2], 4u);
}

TEST(BroadcastingTest, AddScalarBroadcast) {
  // (2, 3) + (1,) -> (2, 3)
  Tensor<float> a(Shape({2, 3}), 2.0f);
  Tensor<float> b(Shape({1}), 3.0f);
  Tensor<float> result = a + b;

  EXPECT_EQ(result.shape()[0], 2u);
  EXPECT_EQ(result.shape()[1], 3u);
  for (size_t i = 0; i < result.numel(); ++i) {
    EXPECT_FLOAT_EQ(result(i), 5.0f);
  }
}

TEST(BroadcastingTest, AddRowBroadcast) {
  // (2, 3) + (3,) -> (2, 3)
  std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> b_data = {10.0f, 20.0f, 30.0f};
  Tensor<float> a(Shape({2, 3}), a_data);
  Tensor<float> b(Shape({3}), b_data);
  Tensor<float> result = a + b;

  EXPECT_EQ(result.shape()[0], 2u);
  EXPECT_EQ(result.shape()[1], 3u);
  EXPECT_FLOAT_EQ(result(0, 0), 11.0f);
  EXPECT_FLOAT_EQ(result(0, 1), 22.0f);
  EXPECT_FLOAT_EQ(result(0, 2), 33.0f);
  EXPECT_FLOAT_EQ(result(1, 0), 14.0f);
  EXPECT_FLOAT_EQ(result(1, 1), 25.0f);
  EXPECT_FLOAT_EQ(result(1, 2), 36.0f);
}

TEST(BroadcastingTest, AddColumnBroadcast) {
  // (2, 3) + (2, 1) -> (2, 3)
  std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> b_data = {10.0f, 20.0f};
  Tensor<float> a(Shape({2, 3}), a_data);
  Tensor<float> b(Shape({2, 1}), b_data);
  Tensor<float> result = a + b;

  EXPECT_EQ(result.shape()[0], 2u);
  EXPECT_EQ(result.shape()[1], 3u);
  EXPECT_FLOAT_EQ(result(0, 0), 11.0f);
  EXPECT_FLOAT_EQ(result(0, 1), 12.0f);
  EXPECT_FLOAT_EQ(result(0, 2), 13.0f);
  EXPECT_FLOAT_EQ(result(1, 0), 24.0f);
  EXPECT_FLOAT_EQ(result(1, 1), 25.0f);
  EXPECT_FLOAT_EQ(result(1, 2), 26.0f);
}

TEST(BroadcastingTest, AddExpandBothDims) {
  // (2, 1) + (1, 3) -> (2, 3)
  std::vector<float> a_data = {1.0f, 2.0f};
  std::vector<float> b_data = {10.0f, 20.0f, 30.0f};
  Tensor<float> a(Shape({2, 1}), a_data);
  Tensor<float> b(Shape({1, 3}), b_data);
  Tensor<float> result = a + b;

  EXPECT_EQ(result.shape()[0], 2u);
  EXPECT_EQ(result.shape()[1], 3u);
  EXPECT_FLOAT_EQ(result(0, 0), 11.0f);
  EXPECT_FLOAT_EQ(result(0, 1), 21.0f);
  EXPECT_FLOAT_EQ(result(0, 2), 31.0f);
  EXPECT_FLOAT_EQ(result(1, 0), 12.0f);
  EXPECT_FLOAT_EQ(result(1, 1), 22.0f);
  EXPECT_FLOAT_EQ(result(1, 2), 32.0f);
}

TEST(BroadcastingTest, SubBroadcast) {
  Tensor<float> a(Shape({2, 3}), 10.0f);
  Tensor<float> b(Shape({3}), 1.0f);
  Tensor<float> result = a - b;

  for (size_t i = 0; i < result.numel(); ++i) {
    EXPECT_FLOAT_EQ(result(i), 9.0f);
  }
}

TEST(BroadcastingTest, MulBroadcast) {
  std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> b_data = {2.0f, 3.0f, 4.0f};
  Tensor<float> a(Shape({2, 3}), a_data);
  Tensor<float> b(Shape({3}), b_data);
  Tensor<float> result = a * b;

  EXPECT_FLOAT_EQ(result(0, 0), 2.0f);
  EXPECT_FLOAT_EQ(result(0, 1), 6.0f);
  EXPECT_FLOAT_EQ(result(0, 2), 12.0f);
  EXPECT_FLOAT_EQ(result(1, 0), 8.0f);
  EXPECT_FLOAT_EQ(result(1, 1), 15.0f);
  EXPECT_FLOAT_EQ(result(1, 2), 24.0f);
}

TEST(BroadcastingTest, DivBroadcast) {
  Tensor<float> a(Shape({2, 3}), 12.0f);
  std::vector<float> b_data = {1.0f, 2.0f, 3.0f};
  Tensor<float> b(Shape({3}), b_data);
  Tensor<float> result = a / b;

  EXPECT_FLOAT_EQ(result(0, 0), 12.0f);
  EXPECT_FLOAT_EQ(result(0, 1), 6.0f);
  EXPECT_FLOAT_EQ(result(0, 2), 4.0f);
  EXPECT_FLOAT_EQ(result(1, 0), 12.0f);
  EXPECT_FLOAT_EQ(result(1, 1), 6.0f);
  EXPECT_FLOAT_EQ(result(1, 2), 4.0f);
}

TEST(BroadcastingTest, Broadcast3D) {
  // (2, 3, 4) + (4,) -> (2, 3, 4)
  Tensor<float> a(Shape({2, 3, 4}), 1.0f);
  std::vector<float> b_data = {1.0f, 2.0f, 3.0f, 4.0f};
  Tensor<float> b(Shape({4}), b_data);
  Tensor<float> result = a + b;

  EXPECT_EQ(result.ndim(), 3u);
  EXPECT_EQ(result.shape()[0], 2u);
  EXPECT_EQ(result.shape()[1], 3u);
  EXPECT_EQ(result.shape()[2], 4u);

  // Check a few values
  EXPECT_FLOAT_EQ(result(0, 0, 0), 2.0f);
  EXPECT_FLOAT_EQ(result(0, 0, 1), 3.0f);
  EXPECT_FLOAT_EQ(result(0, 0, 2), 4.0f);
  EXPECT_FLOAT_EQ(result(0, 0, 3), 5.0f);
}

TEST(BroadcastingTest, Broadcast4DBatchHeads) {
  // Simulating attention mask broadcast: (batch, heads, seq, seq) + (1, 1, seq, seq)
  Tensor<float> a(Shape({2, 4, 3, 3}), 1.0f);
  Tensor<float> mask(Shape({1, 1, 3, 3}), 0.5f);
  Tensor<float> result = a + mask;

  EXPECT_EQ(result.shape()[0], 2u);
  EXPECT_EQ(result.shape()[1], 4u);
  EXPECT_EQ(result.shape()[2], 3u);
  EXPECT_EQ(result.shape()[3], 3u);

  for (size_t i = 0; i < result.numel(); ++i) {
    EXPECT_FLOAT_EQ(result(i), 1.5f);
  }
}

TEST(BroadcastingTest, BroadcastBias) {
  // Simulating bias addition: (batch, seq, hidden) + (hidden,)
  Tensor<float> a(Shape({2, 5, 8}), 1.0f);
  Tensor<float> bias(Shape({8}), 0.1f);
  Tensor<float> result = a + bias;

  EXPECT_EQ(result.shape()[0], 2u);
  EXPECT_EQ(result.shape()[1], 5u);
  EXPECT_EQ(result.shape()[2], 8u);

  for (size_t i = 0; i < result.numel(); ++i) {
    EXPECT_FLOAT_EQ(result(i), 1.1f);
  }
}

TEST(BroadcastingTest, IncompatibleShapesThrow) {
  Tensor<float> a(Shape({2, 3}));
  Tensor<float> b(Shape({4, 3}));
  EXPECT_THROW(a + b, std::runtime_error);
}

TEST(BroadcastingTest, SameShapeFastPath) {
  // Test that same-shape operations still work (fast path)
  Tensor<float> a(Shape({2, 3}), 1.0f);
  Tensor<float> b(Shape({2, 3}), 2.0f);
  Tensor<float> result = a + b;

  EXPECT_EQ(result.shape()[0], 2u);
  EXPECT_EQ(result.shape()[1], 3u);
  for (size_t i = 0; i < result.numel(); ++i) {
    EXPECT_FLOAT_EQ(result(i), 3.0f);
  }
}
