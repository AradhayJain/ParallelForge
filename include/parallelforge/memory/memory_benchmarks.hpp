#pragma once

#include <parallelforge/core/config.hpp>

#include <cstddef>
#include <chrono>

namespace parallelforge::memory {

struct MemoryAccessResult {
    std::size_t working_set_bytes{0};
    std::size_t stride_elements{1};
    std::size_t total_elements_accessed{0};
    std::chrono::nanoseconds elapsed{0};
    double throughput_gbps{0.0};
    double average_latency_ns{0.0};
    uint64_t checksum{0}; // Computed checksum to verify observable work
};

/**
 * @brief Benchmark sequential memory read access over a working set.
 */
[[nodiscard]] MemoryAccessResult benchmark_sequential_access(std::size_t size_bytes, 
                                                            std::size_t repetitions = 5);

/**
 * @brief Benchmark strided memory read access over a working set.
 * Demonstrates cache line utilization drop when stride exceeds cache line size.
 */
[[nodiscard]] MemoryAccessResult benchmark_strided_access(std::size_t size_bytes, 
                                                         std::size_t stride_elements = 16, 
                                                         std::size_t repetitions = 5);

} // namespace parallelforge::memory
