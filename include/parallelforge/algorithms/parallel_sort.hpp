#pragma once

#include <parallelforge/core/config.hpp>
#include <parallelforge/core/types.hpp>

#include <cstddef>
#include <vector>
#include <span>

namespace parallelforge::algorithms {

inline constexpr std::size_t DEFAULT_SORT_CUTOFF = 16384;

/**
 * @brief Serial reference sorting (std::sort wrapper).
 */
void sort_serial(std::span<float> data);

/**
 * @brief Parallel merge sort using std::thread / task partitioning.
 * @param cutoff Threshold below which subarrays are sorted serially.
 * @param max_depth Maximum recursion depth for spawning parallel worker threads.
 */
void sort_parallel(std::span<float> data, 
                   std::size_t cutoff = DEFAULT_SORT_CUTOFF, 
                   unsigned int max_depth = 3);

/**
 * @brief Unified sort dispatcher.
 */
void sort(std::span<float> data, 
          core::AlgorithmType algorithm = core::AlgorithmType::Threads, 
          std::size_t cutoff = DEFAULT_SORT_CUTOFF);

/**
 * @brief Verifies whether data is sorted in ascending order.
 */
[[nodiscard]] bool is_sorted(std::span<const float> data) noexcept;

} // namespace parallelforge::algorithms
