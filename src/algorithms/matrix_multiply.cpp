#include <parallelforge/algorithms/matrix_multiply.hpp>
#include <parallelforge/core/error.hpp>

#include <algorithm>
#include <thread>
#include <vector>

#if PARALLELFORGE_HAS_OPENMP
#include <omp.h>
#endif

#if defined(PARALLELFORGE_ENABLE_AVX2) && defined(__AVX2__)
#include <immintrin.h>
#endif

namespace parallelforge::algorithms {

namespace {

inline void validate_multiplication_operands(const core::Matrix& a, const core::Matrix& b) {
    if (a.cols() != b.rows()) {
        throw core::DimensionMismatchException(
            "Cannot multiply matrix A (" + std::to_string(a.rows()) + "x" +
            std::to_string(a.cols()) + ") by matrix B (" + std::to_string(b.rows()) +
            "x" + std::to_string(b.cols()) + "): inner dimensions must match"
        );
    }
}

inline unsigned int resolve_thread_count(unsigned int requested) noexcept {
    if (requested > 0) {
        return requested;
    }
    const unsigned int hw = std::thread::hardware_concurrency();
    return hw > 0 ? hw : 4;
}

inline bool runtime_supports_avx2() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    #if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    #else
    return false;
    #endif
#else
    return false;
#endif
}

} // namespace

core::Matrix multiply_serial_naive(const core::Matrix& a, const core::Matrix& b) {
    validate_multiplication_operands(a, b);
    const std::size_t M = a.rows();
    const std::size_t K = a.cols();
    const std::size_t N = b.cols();

    core::Matrix c(M, N, 0.0f);
    if (M == 0 || K == 0 || N == 0) {
        return c;
    }

    // Classic i-j-k loop ordering: note that B(k, j) strided access impairs cache spatial locality
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < K; ++k) {
                sum += a(i, k) * b(k, j);
            }
            c(i, j) = sum;
        }
    }
    return c;
}

core::Matrix multiply_serial_ikj(const core::Matrix& a, const core::Matrix& b) {
    validate_multiplication_operands(a, b);
    const std::size_t M = a.rows();
    const std::size_t K = a.cols();
    const std::size_t N = b.cols();

    core::Matrix c(M, N, 0.0f);
    if (M == 0 || K == 0 || N == 0) {
        return c;
    }

    // Cache-friendly i-k-j loop ordering:
    // Both B and C are traversed sequentially in row-major order (stride 1)
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            const float r = a(i, k);
            for (std::size_t j = 0; j < N; ++j) {
                c(i, j) += r * b(k, j);
            }
        }
    }
    return c;
}

core::Matrix multiply_blocked(const core::Matrix& a, const core::Matrix& b, std::size_t tile_size) {
    validate_multiplication_operands(a, b);
    const std::size_t M = a.rows();
    const std::size_t K = a.cols();
    const std::size_t N = b.cols();

    core::Matrix c(M, N, 0.0f);
    if (M == 0 || K == 0 || N == 0) {
        return c;
    }
    if (tile_size == 0) {
        tile_size = core::DEFAULT_TILE_SIZE;
    }

    // 6-deep tiled loop with i-k-j sub-block traversal
    for (std::size_t ii = 0; ii < M; ii += tile_size) {
        const std::size_t i_limit = std::min(ii + tile_size, M);
        for (std::size_t kk = 0; kk < K; kk += tile_size) {
            const std::size_t k_limit = std::min(kk + tile_size, K);
            for (std::size_t jj = 0; jj < N; jj += tile_size) {
                const std::size_t j_limit = std::min(jj + tile_size, N);

                for (std::size_t i = ii; i < i_limit; ++i) {
                    for (std::size_t k = kk; k < k_limit; ++k) {
                        const float r = a(i, k);
                        for (std::size_t j = jj; j < j_limit; ++j) {
                            c(i, j) += r * b(k, j);
                        }
                    }
                }
            }
        }
    }
    return c;
}

core::Matrix multiply_openmp(const core::Matrix& a, 
                             const core::Matrix& b, 
                             std::size_t tile_size,
                             unsigned int threads) {
#if PARALLELFORGE_HAS_OPENMP
    validate_multiplication_operands(a, b);
    const std::size_t M = a.rows();
    const std::size_t K = a.cols();
    const std::size_t N = b.cols();

    core::Matrix c(M, N, 0.0f);
    if (M == 0 || K == 0 || N == 0) {
        return c;
    }
    if (tile_size == 0) {
        tile_size = core::DEFAULT_TILE_SIZE;
    }
    const int num_threads = static_cast<int>(resolve_thread_count(threads));

    // Parallelize outer tile loop over rows of C
    // Disjoint row tiles ensure zero write contention and no synchronization inside tiles
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (std::size_t ii = 0; ii < M; ii += tile_size) {
        const std::size_t i_limit = std::min(ii + tile_size, M);
        for (std::size_t kk = 0; kk < K; kk += tile_size) {
            const std::size_t k_limit = std::min(kk + tile_size, K);
            for (std::size_t jj = 0; jj < N; jj += tile_size) {
                const std::size_t j_limit = std::min(jj + tile_size, N);

                for (std::size_t i = ii; i < i_limit; ++i) {
                    for (std::size_t k = kk; k < k_limit; ++k) {
                        const float r = a(i, k);
                        for (std::size_t j = jj; j < j_limit; ++j) {
                            c(i, j) += r * b(k, j);
                        }
                    }
                }
            }
        }
    }
    return c;
#else
    return multiply_threads(a, b, tile_size, threads);
#endif
}

core::Matrix multiply_threads(const core::Matrix& a, 
                              const core::Matrix& b, 
                              std::size_t tile_size,
                              unsigned int threads) {
    validate_multiplication_operands(a, b);
    const std::size_t M = a.rows();
    const std::size_t K = a.cols();
    const std::size_t N = b.cols();

    core::Matrix c(M, N, 0.0f);
    if (M == 0 || K == 0 || N == 0) {
        return c;
    }
    if (tile_size == 0) {
        tile_size = core::DEFAULT_TILE_SIZE;
    }

    const unsigned int worker_count = std::min(resolve_thread_count(threads), 
                                               static_cast<unsigned int>(std::max<std::size_t>(1, M)));
    if (worker_count <= 1) {
        return multiply_blocked(a, b, tile_size);
    }

    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    // Chunk row-blocks across threads
    const std::size_t total_blocks = (M + tile_size - 1) / tile_size;
    const std::size_t blocks_per_worker = (total_blocks + worker_count - 1) / worker_count;

    for (unsigned int w = 0; w < worker_count; ++w) {
        const std::size_t block_start = w * blocks_per_worker;
        const std::size_t block_end = std::min(block_start + blocks_per_worker, total_blocks);
        if (block_start >= total_blocks) {
            break;
        }

        workers.emplace_back([&a, &b, &c, M, K, N, tile_size, block_start, block_end]() {
            for (std::size_t b_idx = block_start; b_idx < block_end; ++b_idx) {
                const std::size_t ii = b_idx * tile_size;
                const std::size_t i_limit = std::min(ii + tile_size, M);

                for (std::size_t kk = 0; kk < K; kk += tile_size) {
                    const std::size_t k_limit = std::min(kk + tile_size, K);
                    for (std::size_t jj = 0; jj < N; jj += tile_size) {
                        const std::size_t j_limit = std::min(jj + tile_size, N);

                        for (std::size_t i = ii; i < i_limit; ++i) {
                            for (std::size_t k = kk; k < k_limit; ++k) {
                                const float r = a(i, k);
                                for (std::size_t j = jj; j < j_limit; ++j) {
                                    c(i, j) += r * b(k, j);
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    for (auto& t : workers) {
        if (t.joinable()) {
            t.join();
        }
    }

    return c;
}

core::Matrix multiply_simd(const core::Matrix& a, const core::Matrix& b, std::size_t tile_size) {
    validate_multiplication_operands(a, b);
    const std::size_t M = a.rows();
    const std::size_t K = a.cols();
    const std::size_t N = b.cols();

    core::Matrix c(M, N, 0.0f);
    if (M == 0 || K == 0 || N == 0) {
        return c;
    }
    if (tile_size == 0) {
        tile_size = core::DEFAULT_TILE_SIZE;
    }

#if defined(PARALLELFORGE_ENABLE_AVX2) && defined(__AVX2__)
    if (runtime_supports_avx2()) {
        for (std::size_t ii = 0; ii < M; ii += tile_size) {
            const std::size_t i_limit = std::min(ii + tile_size, M);
            for (std::size_t kk = 0; kk < K; kk += tile_size) {
                const std::size_t k_limit = std::min(kk + tile_size, K);
                for (std::size_t jj = 0; jj < N; jj += tile_size) {
                    const std::size_t j_limit = std::min(jj + tile_size, N);

                    for (std::size_t i = ii; i < i_limit; ++i) {
                        for (std::size_t k = kk; k < k_limit; ++k) {
                            const float r_val = a(i, k);
                            const __m256 r_vec = _mm256_set1_ps(r_val);
                            
                            std::size_t j = jj;
                            // Vectorized 8-float AVX2 FMA loop
                            for (; j + 8 <= j_limit; j += 8) {
                                __m256 c_vec = _mm256_loadu_ps(&c(i, j));
                                __m256 b_vec = _mm256_loadu_ps(&b(k, j));
                                c_vec = _mm256_fmadd_ps(r_vec, b_vec, c_vec);
                                _mm256_storeu_ps(&c(i, j), c_vec);
                            }
                            // Remainder tail loop
                            for (; j < j_limit; ++j) {
                                c(i, j) += r_val * b(k, j);
                            }
                        }
                    }
                }
            }
        }
        return c;
    }
#endif

    // High performance fallback with compiler vectorization hints
    return multiply_blocked(a, b, tile_size);
}

core::Matrix multiply(const core::Matrix& a, 
                      const core::Matrix& b, 
                      core::AlgorithmType algorithm,
                      std::size_t tile_size,
                      unsigned int threads) {
    switch (algorithm) {
        case core::AlgorithmType::Serial:
            return multiply_serial_naive(a, b);
        case core::AlgorithmType::Blocked:
            return multiply_blocked(a, b, tile_size);
        case core::AlgorithmType::OpenMP:
            return multiply_openmp(a, b, tile_size, threads);
        case core::AlgorithmType::Threads:
            return multiply_threads(a, b, tile_size, threads);
        case core::AlgorithmType::SIMD:
            return multiply_simd(a, b, tile_size);
    }
    return multiply_blocked(a, b, tile_size);
}

} // namespace parallelforge::algorithms
