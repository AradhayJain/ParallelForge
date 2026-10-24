#pragma once

#include <cstddef>
#include <chrono>

namespace parallelforge::memory {

struct CacheLocalityResult {
    std::size_t matrix_dimension{0};
    std::size_t total_elements{0};
    std::size_t repetitions{0};
    std::chrono::nanoseconds row_major_time{0};
    std::chrono::nanoseconds col_major_time{0};
    double speedup{1.0}; // col_major_time / row_major_time
    float checksum_row{0.0f};
    float checksum_col{0.0f};
};

/**
 * @brief Compares row-major (spatial locality) vs column-major (strided cache thrashing) matrix traversal.
 */
[[nodiscard]] CacheLocalityResult run_cache_locality_experiment(std::size_t dimension = 2048, 
                                                               std::size_t repetitions = 5);

} // namespace parallelforge::memory
