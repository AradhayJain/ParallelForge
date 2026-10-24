#include <gtest/gtest.h>
#include <parallelforge/core/matrix.hpp>
#include <parallelforge/core/error.hpp>

using namespace parallelforge::core;

TEST(MatrixTest, DefaultConstructorIsEmpty) {
    Matrix m;
    EXPECT_EQ(m.rows(), 0);
    EXPECT_EQ(m.cols(), 0);
    EXPECT_EQ(m.size(), 0);
    EXPECT_TRUE(m.empty());
}

TEST(MatrixTest, DimensionsAndIndexing) {
    Matrix m(3, 4, 1.5f);
    EXPECT_EQ(m.rows(), 3);
    EXPECT_EQ(m.cols(), 4);
    EXPECT_EQ(m.size(), 12);
    EXPECT_FALSE(m.empty());

    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 4; ++c) {
            EXPECT_FLOAT_EQ(m(r, c), 1.5f);
            EXPECT_FLOAT_EQ(m.at(r, c), 1.5f);
        }
    }

    m(1, 2) = 42.0f;
    EXPECT_FLOAT_EQ(m.at(1, 2), 42.0f);
}

TEST(MatrixTest, BoundsCheckThrowsException) {
    Matrix m(2, 2, 0.0f);
    EXPECT_THROW((void)m.at(2, 0), DimensionMismatchException);
    EXPECT_THROW((void)m.at(0, 2), DimensionMismatchException);
    EXPECT_THROW((void)m.at(5, 5), DimensionMismatchException);
}

TEST(MatrixTest, RowSpanAccess) {
    Matrix m(2, 3);
    m(0, 0) = 1.0f; m(0, 1) = 2.0f; m(0, 2) = 3.0f;
    m(1, 0) = 4.0f; m(1, 1) = 5.0f; m(1, 2) = 6.0f;

    auto r0 = m.row(0);
    EXPECT_EQ(r0.size(), 3);
    EXPECT_FLOAT_EQ(r0[0], 1.0f);
    EXPECT_FLOAT_EQ(r0[1], 2.0f);
    EXPECT_FLOAT_EQ(r0[2], 3.0f);

    auto r1 = m.row(1);
    EXPECT_EQ(r1.size(), 3);
    EXPECT_FLOAT_EQ(r1[0], 4.0f);
    EXPECT_FLOAT_EQ(r1[1], 5.0f);
    EXPECT_FLOAT_EQ(r1[2], 6.0f);
}

TEST(MatrixTest, Factories) {
    auto z = Matrix::zeros(2, 2);
    EXPECT_FLOAT_EQ(z(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(z(1, 1), 0.0f);

    auto o = Matrix::ones(2, 2);
    EXPECT_FLOAT_EQ(o(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(o(1, 1), 1.0f);

    auto id = Matrix::identity(3);
    EXPECT_FLOAT_EQ(id(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(id(1, 1), 1.0f);
    EXPECT_FLOAT_EQ(id(2, 2), 1.0f);
    EXPECT_FLOAT_EQ(id(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(id(1, 0), 0.0f);

    auto rnd = Matrix::random(10, 10, -5.0f, 5.0f, 123);
    for (std::size_t r = 0; r < 10; ++r) {
        for (std::size_t c = 0; c < 10; ++c) {
            EXPECT_GE(rnd(r, c), -5.0f);
            EXPECT_LE(rnd(r, c), 5.0f);
        }
    }
}

TEST(MatrixTest, ApproxEqualAndTolerance) {
    Matrix m1(2, 2, 1.0f);
    Matrix m2(2, 2, 1.00001f);
    Matrix m3(2, 2, 1.1f);

    float max_diff = 0.0f;
    EXPECT_TRUE(m1.approx_equal(m2, 1e-4f, &max_diff));
    EXPECT_NEAR(max_diff, 0.00001f, 1e-6f);

    EXPECT_FALSE(m1.approx_equal(m3, 1e-4f, &max_diff));
    EXPECT_NEAR(max_diff, 0.1f, 1e-6f);

    Matrix diff_shape(2, 3);
    EXPECT_FALSE(m1.approx_equal(diff_shape));
}

TEST(MatrixTest, OverflowDetection) {
    std::size_t huge = std::numeric_limits<std::size_t>::max() / 2;
    EXPECT_THROW(Matrix(huge, 4), ArithmeticOverflowException);
}
