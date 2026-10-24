#include <parallelforge/memory/memory_benchmarks.hpp>
#include <parallelforge/core/do_not_optimize.hpp>

#include <vector>
#include <numeric>
#include <algorithm>

namespace parallelforge::memory {

MemoryAccessResult benchmark_sequential_access(std::size_t size_bytes, std::size_t repetitions) {
    if (size_bytes < sizeof(int32_t)) {
        size_bytes = sizeof(int32_t);
    }
    const std::size_t num_elements = size_bytes / sizeof(int32_t);
    std::vector<int32_t> buffer(num_elements);

    // Warm-up and initialization
    for (std::size_t i = 0; i < num_elements; ++i) {
        buffer[i] = static_cast<int32_t>(i & 0xFF);
    }
    core::clobber_memory();

    uint64_t total_sum = 0;

    // Warm-up pass
    for (std::size_t i = 0; i < num_elements; ++i) {
        total_sum += static_cast<uint64_t>(buffer[i]);
    }
    core::do_not_optimize(total_sum);

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t rep = 0; rep < repetitions; ++rep) {
        uint64_t sum = 0;
        for (std::size_t i = 0; i < num_elements; ++i) {
            sum += static_cast<uint64_t>(buffer[i]);
        }
        total_sum += sum;
        core::do_not_optimize(sum);
    }
    const auto end = std::chrono::steady_clock::now();
    core::do_not_optimize(total_sum);

    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    const double elapsed_seconds = std::chrono::duration<double>(elapsed).count();
    const double total_bytes = static_cast<double>(num_elements * sizeof(int32_t) * repetitions);

    MemoryAccessResult res;
    res.working_set_bytes = num_elements * sizeof(int32_t);
    res.stride_elements = 1;
    res.total_elements_accessed = num_elements * repetitions;
    res.elapsed = elapsed;
    res.throughput_gbps = elapsed_seconds > 0.0 ? (total_bytes / elapsed_seconds) / 1e9 : 0.0;
    res.average_latency_ns = res.total_elements_accessed > 0 
        ? static_cast<double>(elapsed.count()) / static_cast<double>(res.total_elements_accessed) 
        : 0.0;
    res.checksum = total_sum;

    return res;
}

MemoryAccessResult benchmark_strided_access(std::size_t size_bytes, 
                                           std::size_t stride_elements, 
                                           std::size_t repetitions) {
    if (size_bytes < sizeof(int32_t)) {
        size_bytes = sizeof(int32_t);
    }
    if (stride_elements == 0) {
        stride_elements = 1;
    }
    const std::size_t num_elements = size_bytes / sizeof(int32_t);
    std::vector<int32_t> buffer(num_elements);

    for (std::size_t i = 0; i < num_elements; ++i) {
        buffer[i] = static_cast<int32_t>(i & 0xFF);
    }
    core::clobber_memory();

    uint64_t total_sum = 0;

    // Warm-up pass
    for (std::size_t s = 0; s < stride_elements; ++s) {
        for (std::size_t i = s; i < num_elements; i += stride_elements) {
            total_sum += static_cast<uint64_t>(buffer[i]);
        }
    }
    core::do_not_optimize(total_sum);

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t rep = 0; rep < repetitions; ++rep) {
        uint64_t sum = 0;
        for (std::size_t s = 0; s < stride_elements; ++s) {
            for (std::size_t i = s; i < num_elements; i += stride_elements) {
                sum += static_cast<uint64_t>(buffer[i]);
            }
        }
        total_sum += sum;
        core::do_not_optimize(sum);
    }
    const auto end = std::chrono::steady_clock::now();
    core::do_not_optimize(total_sum);

    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    const double elapsed_seconds = std::chrono::duration<double>(elapsed).count();
    const double total_bytes = static_cast<double>(num_elements * sizeof(int32_t) * repetitions);

    MemoryAccessResult res;
    res.working_set_bytes = num_elements * sizeof(int32_t);
    res.stride_elements = stride_elements;
    res.total_elements_accessed = num_elements * repetitions;
    res.elapsed = elapsed;
    res.throughput_gbps = elapsed_seconds > 0.0 ? (total_bytes / elapsed_seconds) / 1e9 : 0.0;
    res.average_latency_ns = res.total_elements_accessed > 0 
        ? static_cast<double>(elapsed.count()) / static_cast<double>(res.total_elements_accessed) 
        : 0.0;
    res.checksum = total_sum;

    return res;
}

} // namespace parallelforge::memory
