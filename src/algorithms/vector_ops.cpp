#include <parallelforge/algorithms/vector_ops.hpp>
#include <parallelforge/core/error.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
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

inline void validate_vector_sizes(std::size_t a_size, std::size_t b_size) {
    if (a_size != b_size) {
        throw core::DimensionMismatchException(
            "Vector size mismatch: " + std::to_string(a_size) + " != " + std::to_string(b_size)
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

bool float_approx_equal(float a, float b, float tolerance) noexcept {
    const float diff = std::abs(a - b);
    const float scale = std::max({1.0f, std::abs(a), std::abs(b)});
    return diff <= tolerance * scale;
}

bool vector_approx_equal(std::span<const float> a, std::span<const float> b, float tolerance) noexcept {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!float_approx_equal(a[i], b[i], tolerance)) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Vector Addition: c = a + b
// ============================================================================

std::vector<float> vector_add_serial(std::span<const float> a, std::span<const float> b) {
    validate_vector_sizes(a.size(), b.size());
    std::vector<float> c(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        c[i] = a[i] + b[i];
    }
    return c;
}

std::vector<float> vector_add_openmp(std::span<const float> a, std::span<const float> b, unsigned int threads) {
#if PARALLELFORGE_HAS_OPENMP
    validate_vector_sizes(a.size(), b.size());
    std::vector<float> c(a.size());
    const int num_threads = static_cast<int>(resolve_thread_count(threads));
    const auto N = a.size();

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (std::size_t i = 0; i < N; ++i) {
        c[i] = a[i] + b[i];
    }
    return c;
#else
    return vector_add_threads(a, b, threads);
#endif
}

std::vector<float> vector_add_threads(std::span<const float> a, std::span<const float> b, unsigned int threads) {
    validate_vector_sizes(a.size(), b.size());
    const std::size_t N = a.size();
    std::vector<float> c(N);
    if (N == 0) {
        return c;
    }

    const unsigned int worker_count = std::min(resolve_thread_count(threads),
                                               static_cast<unsigned int>(std::max<std::size_t>(1, N / 1024)));
    if (worker_count <= 1) {
        return vector_add_serial(a, b);
    }

    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    const std::size_t chunk = (N + worker_count - 1) / worker_count;

    for (unsigned int w = 0; w < worker_count; ++w) {
        const std::size_t start = w * chunk;
        const std::size_t end = std::min(start + chunk, N);
        if (start >= N) break;

        workers.emplace_back([&a, &b, &c, start, end]() {
            for (std::size_t i = start; i < end; ++i) {
                c[i] = a[i] + b[i];
            }
        });
    }

    for (auto& t : workers) {
        if (t.joinable()) t.join();
    }
    return c;
}

std::vector<float> vector_add_simd(std::span<const float> a, std::span<const float> b) {
    validate_vector_sizes(a.size(), b.size());
    const std::size_t N = a.size();
    std::vector<float> c(N);
    if (N == 0) {
        return c;
    }

#if defined(PARALLELFORGE_ENABLE_AVX2) && defined(__AVX2__)
    if (runtime_supports_avx2()) {
        std::size_t i = 0;
        for (; i + 8 <= N; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            __m256 vc = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(&c[i], vc);
        }
        for (; i < N; ++i) {
            c[i] = a[i] + b[i];
        }
        return c;
    }
#endif

    return vector_add_serial(a, b);
}

// ============================================================================
// Dot Product: sum(a[i] * b[i])
// ============================================================================

float dot_product_serial(std::span<const float> a, std::span<const float> b) {
    validate_vector_sizes(a.size(), b.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float dot_product_openmp(std::span<const float> a, std::span<const float> b, unsigned int threads) {
#if PARALLELFORGE_HAS_OPENMP
    validate_vector_sizes(a.size(), b.size());
    const int num_threads = static_cast<int>(resolve_thread_count(threads));
    const auto N = a.size();
    float sum = 0.0f;

    #pragma omp parallel for num_threads(num_threads) reduction(+:sum) schedule(static)
    for (std::size_t i = 0; i < N; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#else
    return dot_product_threads(a, b, threads);
#endif
}

float dot_product_threads(std::span<const float> a, std::span<const float> b, unsigned int threads) {
    validate_vector_sizes(a.size(), b.size());
    const std::size_t N = a.size();
    if (N == 0) return 0.0f;

    const unsigned int worker_count = std::min(resolve_thread_count(threads),
                                               static_cast<unsigned int>(std::max<std::size_t>(1, N / 1024)));
    if (worker_count <= 1) {
        return dot_product_serial(a, b);
    }

    std::vector<float> partial_sums(worker_count, 0.0f);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    const std::size_t chunk = (N + worker_count - 1) / worker_count;

    for (unsigned int w = 0; w < worker_count; ++w) {
        const std::size_t start = w * chunk;
        const std::size_t end = std::min(start + chunk, N);
        if (start >= N) break;

        workers.emplace_back([&a, &b, &partial_sums, w, start, end]() {
            float local_sum = 0.0f;
            for (std::size_t i = start; i < end; ++i) {
                local_sum += a[i] * b[i];
            }
            partial_sums[w] = local_sum;
        });
    }

    for (auto& t : workers) {
        if (t.joinable()) t.join();
    }

    float total = 0.0f;
    for (float ps : partial_sums) {
        total += ps;
    }
    return total;
}

float dot_product_simd(std::span<const float> a, std::span<const float> b) {
    validate_vector_sizes(a.size(), b.size());
    const std::size_t N = a.size();
    if (N == 0) return 0.0f;

#if defined(PARALLELFORGE_ENABLE_AVX2) && defined(__AVX2__)
    if (runtime_supports_avx2()) {
        __m256 vsum = _mm256_setzero_ps();
        std::size_t i = 0;
        for (; i + 8 <= N; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            vsum = _mm256_fmadd_ps(va, vb, vsum);
        }

        // Horizontal sum of 8 floats in __m256
        alignas(32) float buffer[8];
        _mm256_storeu_ps(buffer, vsum);
        float sum = 0.0f;
        for (float val : buffer) {
            sum += val;
        }

        for (; i < N; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }
#endif

    return dot_product_serial(a, b);
}

// ============================================================================
// Reduction Sum: sum(a[i])
// ============================================================================

float reduce_sum_serial(std::span<const float> a) {
    float sum = 0.0f;
    for (float val : a) {
        sum += val;
    }
    return sum;
}

float reduce_sum_openmp(std::span<const float> a, unsigned int threads) {
#if PARALLELFORGE_HAS_OPENMP
    const int num_threads = static_cast<int>(resolve_thread_count(threads));
    const auto N = a.size();
    float sum = 0.0f;

    #pragma omp parallel for num_threads(num_threads) reduction(+:sum) schedule(static)
    for (std::size_t i = 0; i < N; ++i) {
        sum += a[i];
    }
    return sum;
#else
    return reduce_sum_threads(a, threads);
#endif
}

float reduce_sum_threads(std::span<const float> a, unsigned int threads) {
    const std::size_t N = a.size();
    if (N == 0) return 0.0f;

    const unsigned int worker_count = std::min(resolve_thread_count(threads),
                                               static_cast<unsigned int>(std::max<std::size_t>(1, N / 1024)));
    if (worker_count <= 1) {
        return reduce_sum_serial(a);
    }

    std::vector<float> partial_sums(worker_count, 0.0f);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    const std::size_t chunk = (N + worker_count - 1) / worker_count;

    for (unsigned int w = 0; w < worker_count; ++w) {
        const std::size_t start = w * chunk;
        const std::size_t end = std::min(start + chunk, N);
        if (start >= N) break;

        workers.emplace_back([&a, &partial_sums, w, start, end]() {
            float local_sum = 0.0f;
            for (std::size_t i = start; i < end; ++i) {
                local_sum += a[i];
            }
            partial_sums[w] = local_sum;
        });
    }

    for (auto& t : workers) {
        if (t.joinable()) t.join();
    }

    float total = 0.0f;
    for (float ps : partial_sums) {
        total += ps;
    }
    return total;
}

float reduce_sum_simd(std::span<const float> a) {
    const std::size_t N = a.size();
    if (N == 0) return 0.0f;

#if defined(PARALLELFORGE_ENABLE_AVX2) && defined(__AVX2__)
    if (runtime_supports_avx2()) {
        __m256 vsum = _mm256_setzero_ps();
        std::size_t i = 0;
        for (; i + 8 <= N; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            vsum = _mm256_add_ps(vsum, va);
        }

        alignas(32) float buffer[8];
        _mm256_storeu_ps(buffer, vsum);
        float sum = 0.0f;
        for (float val : buffer) {
            sum += val;
        }

        for (; i < N; ++i) {
            sum += a[i];
        }
        return sum;
    }
#endif

    return reduce_sum_serial(a);
}

// ============================================================================
// SAXPY: y[i] = alpha * x[i] + y[i]
// ============================================================================

void saxpy_serial(float alpha, std::span<const float> x, std::span<float> y) {
    validate_vector_sizes(x.size(), y.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] = alpha * x[i] + y[i];
    }
}

void saxpy_openmp(float alpha, std::span<const float> x, std::span<float> y, unsigned int threads) {
#if PARALLELFORGE_HAS_OPENMP
    validate_vector_sizes(x.size(), y.size());
    const int num_threads = static_cast<int>(resolve_thread_count(threads));
    const auto N = x.size();

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (std::size_t i = 0; i < N; ++i) {
        y[i] = alpha * x[i] + y[i];
    }
#else
    saxpy_threads(alpha, x, y, threads);
#endif
}

void saxpy_threads(float alpha, std::span<const float> x, std::span<float> y, unsigned int threads) {
    validate_vector_sizes(x.size(), y.size());
    const std::size_t N = x.size();
    if (N == 0) return;

    const unsigned int worker_count = std::min(resolve_thread_count(threads),
                                               static_cast<unsigned int>(std::max<std::size_t>(1, N / 1024)));
    if (worker_count <= 1) {
        saxpy_serial(alpha, x, y);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    const std::size_t chunk = (N + worker_count - 1) / worker_count;

    for (unsigned int w = 0; w < worker_count; ++w) {
        const std::size_t start = w * chunk;
        const std::size_t end = std::min(start + chunk, N);
        if (start >= N) break;

        workers.emplace_back([alpha, x, y, start, end]() {
            for (std::size_t i = start; i < end; ++i) {
                y[i] = alpha * x[i] + y[i];
            }
        });
    }

    for (auto& t : workers) {
        if (t.joinable()) t.join();
    }
}

void saxpy_simd(float alpha, std::span<const float> x, std::span<float> y) {
    validate_vector_sizes(x.size(), y.size());
    const std::size_t N = x.size();
    if (N == 0) return;

#if defined(PARALLELFORGE_ENABLE_AVX2) && defined(__AVX2__)
    if (runtime_supports_avx2()) {
        const __m256 valpha = _mm256_set1_ps(alpha);
        std::size_t i = 0;
        for (; i + 8 <= N; i += 8) {
            __m256 vx = _mm256_loadu_ps(&x[i]);
            __m256 vy = _mm256_loadu_ps(&y[i]);
            vy = _mm256_fmadd_ps(valpha, vx, vy);
            _mm256_storeu_ps(&y[i], vy);
        }
        for (; i < N; ++i) {
            y[i] = alpha * x[i] + y[i];
        }
        return;
    }
#endif

    saxpy_serial(alpha, x, y);
}

} // namespace parallelforge::algorithms
