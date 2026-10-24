#pragma once

#include <parallelforge/core/config.hpp>
#include <parallelforge/core/types.hpp>
#include <parallelforge/core/matrix.hpp>

#include <cstddef>
#include <string_view>
#include <optional>

namespace parallelforge::algorithms {

enum class BorderMode {
    ClampToEdge, // Replicate nearest boundary pixel
    ZeroPad,     // Treat out-of-bounds coordinates as 0.0f
    Mirror       // Reflect coordinates across the boundary
};

inline std::string_view to_string(BorderMode mode) noexcept {
    switch (mode) {
        case BorderMode::ClampToEdge: return "clamp";
        case BorderMode::ZeroPad:     return "zero";
        case BorderMode::Mirror:      return "mirror";
    }
    return "unknown";
}

inline std::optional<BorderMode> parse_border_mode(std::string_view sv) noexcept {
    if (sv == "clamp")  return BorderMode::ClampToEdge;
    if (sv == "zero")   return BorderMode::ZeroPad;
    if (sv == "mirror") return BorderMode::Mirror;
    return std::nullopt;
}

// Common standard 2D convolution kernels
struct Kernels {
    static core::Matrix gaussian_blur_3x3();
    static core::Matrix sharpen_3x3();
    static core::Matrix sobel_horizontal_3x3();
    static core::Matrix sobel_vertical_3x3();
    static core::Matrix box_blur_5x5();
};

/**
 * @brief Serial reference implementation of 2D image convolution.
 */
[[nodiscard]] core::Matrix convolve_serial(const core::Matrix& input, 
                                           const core::Matrix& kernel, 
                                           BorderMode border = BorderMode::ClampToEdge);

/**
 * @brief Parallel 2D image convolution using OpenMP (partitioned by rows).
 */
[[nodiscard]] core::Matrix convolve_openmp(const core::Matrix& input, 
                                           const core::Matrix& kernel, 
                                           BorderMode border = BorderMode::ClampToEdge,
                                           unsigned int threads = 0);

/**
 * @brief Parallel 2D image convolution using C++ standard threads (partitioned by rows).
 */
[[nodiscard]] core::Matrix convolve_threads(const core::Matrix& input, 
                                            const core::Matrix& kernel, 
                                            BorderMode border = BorderMode::ClampToEdge,
                                            unsigned int threads = 0);

/**
 * @brief Unified convolution dispatcher.
 */
[[nodiscard]] core::Matrix convolve(const core::Matrix& input, 
                                    const core::Matrix& kernel, 
                                    core::AlgorithmType algorithm = core::AlgorithmType::OpenMP,
                                    BorderMode border = BorderMode::ClampToEdge,
                                    unsigned int threads = 0);

} // namespace parallelforge::algorithms
