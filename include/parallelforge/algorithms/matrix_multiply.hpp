#pragma once

#include <parallelforge/core/config.hpp>
#include <parallelforge/core/types.hpp>
#include <parallelforge/core/matrix.hpp>

#include <cstddef>

namespace parallelforge::algorithms {

/**
 * @brief Serial baseline matrix multiplication (i-j-k naive).
 */
[[nodiscard]] core::Matrix multiply_serial_naive(const core::Matrix& a, const core::Matrix& b);

/**
 * @brief Serial cache-friendly matrix multiplication (i-k-j loop order).
 */
[[nodiscard]] core::Matrix multiply_serial_ikj(const core::Matrix& a, const core::Matrix& b);

/**
 * @brief Cache-blocked / tiled matrix multiplication.
 * @param tile_size Block dimension (fits tiles into L1/L2 cache).
 */
[[nodiscard]] core::Matrix multiply_blocked(const core::Matrix& a, 
                                            const core::Matrix& b, 
                                            std::size_t tile_size = core::DEFAULT_TILE_SIZE);

/**
 * @brief Parallel blocked matrix multiplication using OpenMP.
 */
[[nodiscard]] core::Matrix multiply_openmp(const core::Matrix& a, 
                                           const core::Matrix& b, 
                                           std::size_t tile_size = core::DEFAULT_TILE_SIZE,
                                           unsigned int threads = 0);

/**
 * @brief Parallel blocked matrix multiplication using C++ standard threads.
 */
[[nodiscard]] core::Matrix multiply_threads(const core::Matrix& a, 
                                            const core::Matrix& b, 
                                            std::size_t tile_size = core::DEFAULT_TILE_SIZE,
                                            unsigned int threads = 0);

/**
 * @brief SIMD / vector-optimized matrix multiplication.
 */
[[nodiscard]] core::Matrix multiply_simd(const core::Matrix& a, 
                                         const core::Matrix& b, 
                                         std::size_t tile_size = core::DEFAULT_TILE_SIZE);

/**
 * @brief Unified dispatcher for matrix multiplication.
 */
[[nodiscard]] core::Matrix multiply(const core::Matrix& a, 
                                    const core::Matrix& b, 
                                    core::AlgorithmType algorithm = core::AlgorithmType::Blocked,
                                    std::size_t tile_size = core::DEFAULT_TILE_SIZE,
                                    unsigned int threads = 0);

} // namespace parallelforge::algorithms
