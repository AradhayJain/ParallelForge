#pragma once

#include <parallelforge/core/config.hpp>

#include <cstddef>
#include <chrono>
#include <cstdint>

namespace parallelforge::memory {

struct FalseSharingResult {
    unsigned int threads{1};
    std::size_t iterations_per_thread{0};
    std::chrono::nanoseconds unpadded_time{0};
    std::chrono::nanoseconds padded_time{0};
    double speedup{1.0}; // unpadded_time / padded_time
    uint64_t total_increments{0};
};

/**
 * @brief Demonstrates the cache coherence penalty of false sharing vs cache-line padding.
 */
[[nodiscard]] FalseSharingResult run_false_sharing_experiment(unsigned int threads = 4, 
                                                             std::size_t iterations = 20'000'000);

} // namespace parallelforge::memory
