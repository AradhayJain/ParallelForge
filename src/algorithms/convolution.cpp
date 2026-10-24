#include <parallelforge/algorithms/convolution.hpp>
#include <parallelforge/core/error.hpp>

#include <algorithm>
#include <thread>
#include <vector>

#if PARALLELFORGE_HAS_OPENMP
#include <omp.h>
#endif

namespace parallelforge::algorithms {

namespace {

inline void validate_convolution_inputs(const core::Matrix& input, const core::Matrix& kernel) {
    if (kernel.rows() == 0 || kernel.cols() == 0) {
        throw core::InvalidConfigurationException("Convolution kernel cannot be empty");
    }
    if (kernel.rows() % 2 == 0 || kernel.cols() % 2 == 0) {
        throw core::InvalidConfigurationException(
            "Convolution kernel dimensions must be odd (received " +
            std::to_string(kernel.rows()) + "x" + std::to_string(kernel.cols()) + ")"
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

inline float sample_pixel(const core::Matrix& input, 
                          long long y, 
                          long long x, 
                          BorderMode border) noexcept {
    const auto H = static_cast<long long>(input.rows());
    const auto W = static_cast<long long>(input.cols());

    if (x >= 0 && x < W && y >= 0 && y < H) {
        return input(static_cast<std::size_t>(y), static_cast<std::size_t>(x));
    }

    switch (border) {
        case BorderMode::ZeroPad:
            return 0.0f;

        case BorderMode::ClampToEdge: {
            const long long clamped_y = std::clamp(y, 0LL, H - 1);
            const long long clamped_x = std::clamp(x, 0LL, W - 1);
            return input(static_cast<std::size_t>(clamped_y), static_cast<std::size_t>(clamped_x));
        }

        case BorderMode::Mirror: {
            long long my = y;
            if (my < 0) {
                my = -my;
            } else if (my >= H) {
                my = 2 * (H - 1) - my;
            }
            my = std::clamp(my, 0LL, H - 1);

            long long mx = x;
            if (mx < 0) {
                mx = -mx;
            } else if (mx >= W) {
                mx = 2 * (W - 1) - mx;
            }
            mx = std::clamp(mx, 0LL, W - 1);

            return input(static_cast<std::size_t>(my), static_cast<std::size_t>(mx));
        }
    }
    return 0.0f;
}

void convolve_row_range(const core::Matrix& input, 
                        const core::Matrix& kernel, 
                        core::Matrix& output, 
                        BorderMode border, 
                        std::size_t start_row, 
                        std::size_t end_row) {
    const std::size_t W = input.cols();
    const auto kh = static_cast<long long>(kernel.rows());
    const auto kw = static_cast<long long>(kernel.cols());
    const long long radius_y = kh / 2;
    const long long radius_x = kw / 2;

    for (std::size_t y = start_row; y < end_row; ++y) {
        for (std::size_t x = 0; x < W; ++x) {
            float sum = 0.0f;
            for (long long ky = 0; ky < kh; ++ky) {
                const long long sample_y = static_cast<long long>(y) + ky - radius_y;
                for (long long kx = 0; kx < kw; ++kx) {
                    const long long sample_x = static_cast<long long>(x) + kx - radius_x;
                    const float k_val = kernel(static_cast<std::size_t>(ky), static_cast<std::size_t>(kx));
                    sum += k_val * sample_pixel(input, sample_y, sample_x, border);
                }
            }
            output(y, x) = sum;
        }
    }
}

} // namespace

core::Matrix Kernels::gaussian_blur_3x3() {
    return core::Matrix(3, 3, std::vector<float>{
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
        2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f
    });
}

core::Matrix Kernels::sharpen_3x3() {
    return core::Matrix(3, 3, std::vector<float>{
         0.0f, -1.0f,  0.0f,
        -1.0f,  5.0f, -1.0f,
         0.0f, -1.0f,  0.0f
    });
}

core::Matrix Kernels::sobel_horizontal_3x3() {
    return core::Matrix(3, 3, std::vector<float>{
        -1.0f, 0.0f, 1.0f,
        -2.0f, 0.0f, 2.0f,
        -1.0f, 0.0f, 1.0f
    });
}

core::Matrix Kernels::sobel_vertical_3x3() {
    return core::Matrix(3, 3, std::vector<float>{
        -1.0f, -2.0f, -1.0f,
         0.0f,  0.0f,  0.0f,
         1.0f,  2.0f,  1.0f
    });
}

core::Matrix Kernels::box_blur_5x5() {
    return core::Matrix(5, 5, 1.0f / 25.0f);
}

core::Matrix convolve_serial(const core::Matrix& input, 
                            const core::Matrix& kernel, 
                            BorderMode border) {
    validate_convolution_inputs(input, kernel);
    const std::size_t H = input.rows();
    const std::size_t W = input.cols();

    core::Matrix output(H, W, 0.0f);
    if (H == 0 || W == 0) {
        return output;
    }

    convolve_row_range(input, kernel, output, border, 0, H);
    return output;
}

core::Matrix convolve_openmp(const core::Matrix& input, 
                            const core::Matrix& kernel, 
                            BorderMode border,
                            unsigned int threads) {
#if PARALLELFORGE_HAS_OPENMP
    validate_convolution_inputs(input, kernel);
    const std::size_t H = input.rows();
    const std::size_t W = input.cols();

    core::Matrix output(H, W, 0.0f);
    if (H == 0 || W == 0) {
        return output;
    }

    const int num_threads = static_cast<int>(resolve_thread_count(threads));
    const auto kh = static_cast<long long>(kernel.rows());
    const auto kw = static_cast<long long>(kernel.cols());
    const long long radius_y = kh / 2;
    const long long radius_x = kw / 2;

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (std::size_t y = 0; y < H; ++y) {
        for (std::size_t x = 0; x < W; ++x) {
            float sum = 0.0f;
            for (long long ky = 0; ky < kh; ++ky) {
                const long long sample_y = static_cast<long long>(y) + ky - radius_y;
                for (long long kx = 0; kx < kw; ++kx) {
                    const long long sample_x = static_cast<long long>(x) + kx - radius_x;
                    const float k_val = kernel(static_cast<std::size_t>(ky), static_cast<std::size_t>(kx));
                    sum += k_val * sample_pixel(input, sample_y, sample_x, border);
                }
            }
            output(y, x) = sum;
        }
    }
    return output;
#else
    return convolve_threads(input, kernel, border, threads);
#endif
}

core::Matrix convolve_threads(const core::Matrix& input, 
                             const core::Matrix& kernel, 
                             BorderMode border,
                             unsigned int threads) {
    validate_convolution_inputs(input, kernel);
    const std::size_t H = input.rows();
    const std::size_t W = input.cols();

    core::Matrix output(H, W, 0.0f);
    if (H == 0 || W == 0) {
        return output;
    }

    const unsigned int worker_count = std::min(resolve_thread_count(threads),
                                               static_cast<unsigned int>(std::max<std::size_t>(1, H)));
    if (worker_count <= 1) {
        convolve_row_range(input, kernel, output, border, 0, H);
        return output;
    }

    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    const std::size_t rows_per_worker = (H + worker_count - 1) / worker_count;

    for (unsigned int w = 0; w < worker_count; ++w) {
        const std::size_t start_row = w * rows_per_worker;
        const std::size_t end_row = std::min(start_row + rows_per_worker, H);
        if (start_row >= H) {
            break;
        }

        workers.emplace_back([&input, &kernel, &output, border, start_row, end_row]() {
            convolve_row_range(input, kernel, output, border, start_row, end_row);
        });
    }

    for (auto& t : workers) {
        if (t.joinable()) {
            t.join();
        }
    }

    return output;
}

core::Matrix convolve(const core::Matrix& input, 
                     const core::Matrix& kernel, 
                     core::AlgorithmType algorithm,
                     BorderMode border,
                     unsigned int threads) {
    switch (algorithm) {
        case core::AlgorithmType::Serial:
            return convolve_serial(input, kernel, border);
        case core::AlgorithmType::OpenMP:
            return convolve_openmp(input, kernel, border, threads);
        case core::AlgorithmType::Threads:
        case core::AlgorithmType::Blocked:
        case core::AlgorithmType::SIMD:
            return convolve_threads(input, kernel, border, threads);
    }
    return convolve_serial(input, kernel, border);
}

} // namespace parallelforge::algorithms
