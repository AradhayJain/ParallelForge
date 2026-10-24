#include <parallelforge/algorithms/parallel_sort.hpp>

#include <algorithm>
#include <thread>
#include <vector>

namespace parallelforge::algorithms {

namespace {

void parallel_merge_sort_recursive(std::span<float> data, 
                                  std::span<float> scratch, 
                                  std::size_t cutoff, 
                                  unsigned int depth, 
                                  unsigned int max_depth) {
    const std::size_t n = data.size();
    if (n <= cutoff || depth >= max_depth) {
        std::sort(data.begin(), data.end());
        return;
    }

    const std::size_t mid = n / 2;
    auto left_data = data.subspan(0, mid);
    auto right_data = data.subspan(mid);
    auto left_scratch = scratch.subspan(0, mid);
    auto right_scratch = scratch.subspan(mid);

    // Fork left partition into worker thread, run right partition on calling thread
    std::thread worker([left_data, left_scratch, cutoff, depth, max_depth]() mutable {
        parallel_merge_sort_recursive(left_data, left_scratch, cutoff, depth + 1, max_depth);
    });

    parallel_merge_sort_recursive(right_data, right_scratch, cutoff, depth + 1, max_depth);
    worker.join();

    // Cache-friendly merge into scratch buffer, then copy back
    std::size_t i = 0;
    std::size_t j = mid;
    std::size_t k = 0;

    while (i < mid && j < n) {
        if (data[i] <= data[j]) {
            scratch[k++] = data[i++];
        } else {
            scratch[k++] = data[j++];
        }
    }
    while (i < mid) {
        scratch[k++] = data[i++];
    }
    while (j < n) {
        scratch[k++] = data[j++];
    }

    std::copy_n(scratch.data(), n, data.data());
}

} // namespace

void sort_serial(std::span<float> data) {
    std::sort(data.begin(), data.end());
}

void sort_parallel(std::span<float> data, std::size_t cutoff, unsigned int max_depth) {
    if (data.size() <= cutoff) {
        sort_serial(data);
        return;
    }

    // Allocate scratch merge buffer once at the top level to avoid per-recursion allocations
    std::vector<float> scratch(data.size());
    parallel_merge_sort_recursive(data, scratch, cutoff, 0, max_depth);
}

void sort(std::span<float> data, core::AlgorithmType algorithm, std::size_t cutoff) {
    switch (algorithm) {
        case core::AlgorithmType::Serial:
            sort_serial(data);
            break;
        case core::AlgorithmType::Threads:
        case core::AlgorithmType::OpenMP:
        case core::AlgorithmType::Blocked:
        case core::AlgorithmType::SIMD:
            sort_parallel(data, cutoff);
            break;
    }
}

bool is_sorted(std::span<const float> data) noexcept {
    return std::is_sorted(data.begin(), data.end());
}

} // namespace parallelforge::algorithms
