#pragma once

#include <parallelforge/core/config.hpp>
#include <parallelforge/core/types.hpp>

#include <cstddef>
#include <vector>
#include <span>

namespace parallelforge::algorithms {

/**
 * @brief Vector addition: c = a + b
 */
[[nodiscard]] std::vector<float> vector_add_serial(std::span<const float> a, std::span<const float> b);
[[nodiscard]] std::vector<float> vector_add_openmp(std::span<const float> a, std::span<const float> b, unsigned int threads = 0);
[[nodiscard]] std::vector<float> vector_add_threads(std::span<const float> a, std::span<const float> b, unsigned int threads = 0);
[[nodiscard]] std::vector<float> vector_add_simd(std::span<const float> a, std::span<const float> b);

/**
 * @brief Dot product: result = sum(a[i] * b[i])
 */
[[nodiscard]] float dot_product_serial(std::span<const float> a, std::span<const float> b);
[[nodiscard]] float dot_product_openmp(std::span<const float> a, std::span<const float> b, unsigned int threads = 0);
[[nodiscard]] float dot_product_threads(std::span<const float> a, std::span<const float> b, unsigned int threads = 0);
[[nodiscard]] float dot_product_simd(std::span<const float> a, std::span<const float> b);

/**
 * @brief Reduction / sum: result = sum(a[i])
 */
[[nodiscard]] float reduce_sum_serial(std::span<const float> a);
[[nodiscard]] float reduce_sum_openmp(std::span<const float> a, unsigned int threads = 0);
[[nodiscard]] float reduce_sum_threads(std::span<const float> a, unsigned int threads = 0);
[[nodiscard]] float reduce_sum_simd(std::span<const float> a);

/**
 * @brief Element-wise transformation (SAXPY): y[i] = alpha * x[i] + y[i]
 */
void saxpy_serial(float alpha, std::span<const float> x, std::span<float> y);
void saxpy_openmp(float alpha, std::span<const float> x, std::span<float> y, unsigned int threads = 0);
void saxpy_threads(float alpha, std::span<const float> x, std::span<float> y, unsigned int threads = 0);
void saxpy_simd(float alpha, std::span<const float> x, std::span<float> y);

/**
 * @brief Tolerance check helper for floating point results.
 */
[[nodiscard]] bool float_approx_equal(float a, float b, float tolerance = core::FLOAT_EPSILON) noexcept;
[[nodiscard]] bool vector_approx_equal(std::span<const float> a, std::span<const float> b, float tolerance = core::FLOAT_EPSILON) noexcept;

} // namespace parallelforge::algorithms
