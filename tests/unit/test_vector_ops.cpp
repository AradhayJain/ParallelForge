#include <gtest/gtest.h>
#include <parallelforge/algorithms/vector_ops.hpp>
#include <parallelforge/core/error.hpp>

using namespace parallelforge::core;
using namespace parallelforge::algorithms;

TEST(VectorOpsTest, DimensionMismatchThrows) {
    std::vector<float> a(10, 1.0f);
    std::vector<float> b(15, 2.0f);
    EXPECT_THROW(vector_add_serial(a, b), DimensionMismatchException);
    EXPECT_THROW(dot_product_serial(a, b), DimensionMismatchException);
    EXPECT_THROW(saxpy_serial(2.0f, a, b), DimensionMismatchException);
}

TEST(VectorOpsTest, EmptyVectorsHandledSafely) {
    std::vector<float> empty_a;
    std::vector<float> empty_b;

    auto add_res = vector_add_serial(empty_a, empty_b);
    EXPECT_TRUE(add_res.empty());

    float dot_res = dot_product_serial(empty_a, empty_b);
    EXPECT_FLOAT_EQ(dot_res, 0.0f);

    float sum_res = reduce_sum_serial(empty_a);
    EXPECT_FLOAT_EQ(sum_res, 0.0f);

    saxpy_serial(2.0f, empty_a, empty_b);
    EXPECT_TRUE(empty_b.empty());
}

TEST(VectorOpsTest, VectorAdditionCrossValidation) {
    const std::size_t N = 100000;
    std::vector<float> a(N, 1.75f);
    std::vector<float> b(N, 2.25f);

    auto serial = vector_add_serial(a, b);
    auto omp = vector_add_openmp(a, b, 4);
    auto th = vector_add_threads(a, b, 4);
    auto simd = vector_add_simd(a, b);

    EXPECT_TRUE(vector_approx_equal(serial, omp));
    EXPECT_TRUE(vector_approx_equal(serial, th));
    EXPECT_TRUE(vector_approx_equal(serial, simd));
    EXPECT_FLOAT_EQ(serial[0], 4.0f);
}

TEST(VectorOpsTest, DotProductCrossValidation) {
    const std::size_t N = 50000;
    std::vector<float> a(N, 0.5f);
    std::vector<float> b(N, 4.0f);

    float serial = dot_product_serial(a, b);
    float omp = dot_product_openmp(a, b, 4);
    float th = dot_product_threads(a, b, 4);
    float simd = dot_product_simd(a, b);

    const float expected = static_cast<float>(N) * 2.0f;
    EXPECT_TRUE(float_approx_equal(serial, expected));
    EXPECT_TRUE(float_approx_equal(serial, omp));
    EXPECT_TRUE(float_approx_equal(serial, th));
    EXPECT_TRUE(float_approx_equal(serial, simd));
}

TEST(VectorOpsTest, ReduceSumCrossValidation) {
    const std::size_t N = 100000;
    std::vector<float> a(N, 1.0f);

    float serial = reduce_sum_serial(a);
    float omp = reduce_sum_openmp(a, 4);
    float th = reduce_sum_threads(a, 4);
    float simd = reduce_sum_simd(a);

    EXPECT_TRUE(float_approx_equal(serial, static_cast<float>(N)));
    EXPECT_TRUE(float_approx_equal(serial, omp));
    EXPECT_TRUE(float_approx_equal(serial, th));
    EXPECT_TRUE(float_approx_equal(serial, simd));
}

TEST(VectorOpsTest, SaxpyCrossValidation) {
    const std::size_t N = 50000;
    std::vector<float> x(N, 2.0f);
    std::vector<float> y1(N, 1.0f);
    std::vector<float> y2(N, 1.0f);
    std::vector<float> y3(N, 1.0f);
    std::vector<float> y4(N, 1.0f);

    saxpy_serial(3.0f, x, y1);
    saxpy_openmp(3.0f, x, y2, 4);
    saxpy_threads(3.0f, x, y3, 4);
    saxpy_simd(3.0f, x, y4);

    EXPECT_FLOAT_EQ(y1[0], 7.0f);
    EXPECT_TRUE(vector_approx_equal(y1, y2));
    EXPECT_TRUE(vector_approx_equal(y1, y3));
    EXPECT_TRUE(vector_approx_equal(y1, y4));
}
