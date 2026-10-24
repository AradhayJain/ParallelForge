#include <gtest/gtest.h>
#include <parallelforge/algorithms/matrix_multiply.hpp>
#include <parallelforge/core/matrix.hpp>
#include <parallelforge/core/error.hpp>

using namespace parallelforge::core;
using namespace parallelforge::algorithms;

TEST(MatrixMultiplyTest, IncompatibleDimensionsThrow) {
    Matrix a(3, 4);
    Matrix b(5, 6);
    EXPECT_THROW(multiply_serial_naive(a, b), DimensionMismatchException);
    EXPECT_THROW(multiply_blocked(a, b), DimensionMismatchException);
    EXPECT_THROW(multiply_threads(a, b), DimensionMismatchException);
}

TEST(MatrixMultiplyTest, ZeroDimensionsPolicy) {
    Matrix a(0, 5);
    Matrix b(5, 4);
    Matrix c = multiply_blocked(a, b);
    EXPECT_EQ(c.rows(), 0);
    EXPECT_EQ(c.cols(), 4);
    EXPECT_TRUE(c.empty());

    Matrix a2(3, 0);
    Matrix b2(0, 4);
    Matrix c2 = multiply_blocked(a2, b2);
    EXPECT_EQ(c2.rows(), 3);
    EXPECT_EQ(c2.cols(), 4);
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t col = 0; col < 4; ++col) {
            EXPECT_FLOAT_EQ(c2(r, col), 0.0f);
        }
    }
}

TEST(MatrixMultiplyTest, IdentityMultiplication) {
    Matrix a = Matrix::random(16, 16, -10.0f, 10.0f, 42);
    Matrix id = Matrix::identity(16);

    Matrix c_serial = multiply_serial_ikj(a, id);
    EXPECT_TRUE(a.approx_equal(c_serial));

    Matrix c_blocked = multiply_blocked(a, id, 8);
    EXPECT_TRUE(a.approx_equal(c_blocked));
}

TEST(MatrixMultiplyTest, KnownSmallMultiplication) {
    // [1 2] * [5 6] = [19 22]
    // [3 4]   [7 8]   [43 50]
    Matrix a(2, 2, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    Matrix b(2, 2, std::vector<float>{5.0f, 6.0f, 7.0f, 8.0f});
    Matrix expected(2, 2, std::vector<float>{19.0f, 22.0f, 43.0f, 50.0f});

    Matrix c_naive = multiply_serial_naive(a, b);
    EXPECT_TRUE(c_naive.approx_equal(expected));

    Matrix c_ikj = multiply_serial_ikj(a, b);
    EXPECT_TRUE(c_ikj.approx_equal(expected));

    Matrix c_blocked = multiply_blocked(a, b, 1);
    EXPECT_TRUE(c_blocked.approx_equal(expected));

    Matrix c_threads = multiply_threads(a, b, 1, 2);
    EXPECT_TRUE(c_threads.approx_equal(expected));
}

TEST(MatrixMultiplyTest, CrossValidationAdversarialDimensions) {
    // Non-power of 2, non-tile-aligned rectangular matrices
    const std::size_t M = 67;
    const std::size_t K = 129;
    const std::size_t N = 93;
    const std::size_t tile_size = 32;

    Matrix a = Matrix::random(M, K, -2.0f, 2.0f, 111);
    Matrix b = Matrix::random(K, N, -2.0f, 2.0f, 222);

    Matrix ref = multiply_serial_ikj(a, b);
    Matrix naive = multiply_serial_naive(a, b);
    EXPECT_TRUE(ref.approx_equal(naive));

    Matrix blocked = multiply_blocked(a, b, tile_size);
    EXPECT_TRUE(ref.approx_equal(blocked));

    Matrix omp = multiply_openmp(a, b, tile_size, 4);
    EXPECT_TRUE(ref.approx_equal(omp));

    Matrix th = multiply_threads(a, b, tile_size, 4);
    EXPECT_TRUE(ref.approx_equal(th));

    Matrix simd = multiply_simd(a, b, tile_size);
    EXPECT_TRUE(ref.approx_equal(simd));
}
