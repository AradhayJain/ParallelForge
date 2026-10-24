#include <parallelforge/memory/cache_locality.hpp>
#include <parallelforge/core/matrix.hpp>
#include <parallelforge/core/do_not_optimize.hpp>

namespace parallelforge::memory {

CacheLocalityResult run_cache_locality_experiment(std::size_t dimension, std::size_t repetitions) {
    if (dimension == 0) dimension = 1024;
    if (repetitions == 0) repetitions = 1;

    core::Matrix m = core::Matrix::random(dimension, dimension, 0.0f, 10.0f, 12345);
    core::clobber_memory();

    // Warm-up pass
    float warm_sum = 0.0f;
    for (std::size_t r = 0; r < dimension; ++r) {
        for (std::size_t c = 0; c < dimension; ++c) {
            warm_sum += m(r, c);
        }
    }
    core::do_not_optimize(warm_sum);

    // 1. Measure Row-major traversal (Stride 1, contiguous sequential access)
    float row_sum = 0.0f;
    const auto start_row = std::chrono::steady_clock::now();
    for (std::size_t rep = 0; rep < repetitions; ++rep) {
        float rep_sum = 0.0f;
        for (std::size_t r = 0; r < dimension; ++r) {
            for (std::size_t c = 0; c < dimension; ++c) {
                rep_sum += m(r, c);
            }
        }
        row_sum += rep_sum;
        core::do_not_optimize(rep_sum);
    }
    const auto end_row = std::chrono::steady_clock::now();
    core::do_not_optimize(row_sum);

    // 2. Measure Column-major traversal (Stride N * sizeof(float), jumping cache lines)
    float col_sum = 0.0f;
    const auto start_col = std::chrono::steady_clock::now();
    for (std::size_t rep = 0; rep < repetitions; ++rep) {
        float rep_sum = 0.0f;
        for (std::size_t c = 0; c < dimension; ++c) {
            for (std::size_t r = 0; r < dimension; ++r) {
                rep_sum += m(r, c);
            }
        }
        col_sum += rep_sum;
        core::do_not_optimize(rep_sum);
    }
    const auto end_col = std::chrono::steady_clock::now();
    core::do_not_optimize(col_sum);

    const auto row_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end_row - start_row);
    const auto col_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end_col - start_col);

    CacheLocalityResult res;
    res.matrix_dimension = dimension;
    res.total_elements = dimension * dimension * repetitions;
    res.repetitions = repetitions;
    res.row_major_time = row_elapsed;
    res.col_major_time = col_elapsed;
    res.speedup = row_elapsed.count() > 0 
        ? static_cast<double>(col_elapsed.count()) / static_cast<double>(row_elapsed.count()) 
        : 1.0;
    res.checksum_row = row_sum;
    res.checksum_col = col_sum;

    return res;
}

} // namespace parallelforge::memory
