#include <gtest/gtest.h>
#include <parallelforge/algorithms/convolution.hpp>
#include <parallelforge/core/matrix.hpp>
#include <parallelforge/core/error.hpp>

using namespace parallelforge::core;
using namespace parallelforge::algorithms;

TEST(ConvolutionTest, InvalidKernelDimensionsThrow) {
    Matrix img(10, 10, 1.0f);
    Matrix even_kernel(2, 2, 1.0f);
    EXPECT_THROW(convolve_serial(img, even_kernel), InvalidConfigurationException);

    Matrix empty_kernel;
    EXPECT_THROW(convolve_serial(img, empty_kernel), InvalidConfigurationException);
}

TEST(ConvolutionTest, IdentityKernelPreservesImage) {
    Matrix img = Matrix::random(32, 32, 0.0f, 255.0f, 42);
    // 3x3 identity kernel: 1.0 in center, 0 elsewhere
    Matrix identity_kernel(3, 3, std::vector<float>{
        0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f
    });

    Matrix out_serial = convolve_serial(img, identity_kernel, BorderMode::ClampToEdge);
    EXPECT_TRUE(img.approx_equal(out_serial));

    Matrix out_omp = convolve_openmp(img, identity_kernel, BorderMode::ClampToEdge, 4);
    EXPECT_TRUE(img.approx_equal(out_omp));

    Matrix out_th = convolve_threads(img, identity_kernel, BorderMode::ClampToEdge, 4);
    EXPECT_TRUE(img.approx_equal(out_th));
}

TEST(ConvolutionTest, BorderModesCorrectness) {
    // 3x3 image with known values
    Matrix img(3, 3, std::vector<float>{
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    });

    Matrix box = Kernels::box_blur_5x5();

    Matrix serial_clamp = convolve_serial(img, box, BorderMode::ClampToEdge);
    Matrix omp_clamp = convolve_openmp(img, box, BorderMode::ClampToEdge, 2);
    Matrix th_clamp = convolve_threads(img, box, BorderMode::ClampToEdge, 2);
    EXPECT_TRUE(serial_clamp.approx_equal(omp_clamp));
    EXPECT_TRUE(serial_clamp.approx_equal(th_clamp));

    Matrix serial_zero = convolve_serial(img, box, BorderMode::ZeroPad);
    Matrix omp_zero = convolve_openmp(img, box, BorderMode::ZeroPad, 2);
    Matrix th_zero = convolve_threads(img, box, BorderMode::ZeroPad, 2);
    EXPECT_TRUE(serial_zero.approx_equal(omp_zero));
    EXPECT_TRUE(serial_zero.approx_equal(th_zero));

    Matrix serial_mirror = convolve_serial(img, box, BorderMode::Mirror);
    Matrix omp_mirror = convolve_openmp(img, box, BorderMode::Mirror, 2);
    Matrix th_mirror = convolve_threads(img, box, BorderMode::Mirror, 2);
    EXPECT_TRUE(serial_mirror.approx_equal(omp_mirror));
    EXPECT_TRUE(serial_mirror.approx_equal(th_mirror));
}

TEST(ConvolutionTest, CrossValidationLargeImage) {
    Matrix img = Matrix::random(128, 256, 0.0f, 255.0f, 999);
    Matrix kernel = Kernels::gaussian_blur_3x3();

    Matrix ref = convolve_serial(img, kernel, BorderMode::ClampToEdge);
    Matrix omp = convolve_openmp(img, kernel, BorderMode::ClampToEdge, 4);
    Matrix th = convolve_threads(img, kernel, BorderMode::ClampToEdge, 4);

    EXPECT_TRUE(ref.approx_equal(omp));
    EXPECT_TRUE(ref.approx_equal(th));
}
