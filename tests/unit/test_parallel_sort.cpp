#include <gtest/gtest.h>
#include <parallelforge/algorithms/parallel_sort.hpp>

#include <vector>
#include <random>

using namespace parallelforge::algorithms;

TEST(ParallelSortTest, EdgeCases) {
    std::vector<float> empty_vec;
    sort_parallel(empty_vec);
    EXPECT_TRUE(is_sorted(empty_vec));

    std::vector<float> single = {42.0f};
    sort_parallel(single);
    EXPECT_TRUE(is_sorted(single));

    std::vector<float> pair = {5.0f, 2.0f};
    sort_parallel(pair);
    EXPECT_TRUE(is_sorted(pair));
    EXPECT_FLOAT_EQ(pair[0], 2.0f);
    EXPECT_FLOAT_EQ(pair[1], 5.0f);
}

TEST(ParallelSortTest, AlreadySortedAndReverseSorted) {
    std::vector<float> sorted_arr(1000);
    for (std::size_t i = 0; i < 1000; ++i) sorted_arr[i] = static_cast<float>(i);
    sort_parallel(sorted_arr, 100);
    EXPECT_TRUE(is_sorted(sorted_arr));

    std::vector<float> rev_arr(1000);
    for (std::size_t i = 0; i < 1000; ++i) rev_arr[i] = static_cast<float>(1000 - i);
    sort_parallel(rev_arr, 100);
    EXPECT_TRUE(is_sorted(rev_arr));
    EXPECT_FLOAT_EQ(rev_arr[0], 1.0f);
    EXPECT_FLOAT_EQ(rev_arr[999], 1000.0f);
}

TEST(ParallelSortTest, LargeRandomArrayMatchesSerial) {
    const std::size_t N = 100000;
    std::vector<float> a(N);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10000.0f, 10000.0f);
    for (auto& val : a) val = dist(gen);

    std::vector<float> b = a;

    sort_serial(a);
    EXPECT_TRUE(is_sorted(a));

    sort_parallel(b, 2048, 4);
    EXPECT_TRUE(is_sorted(b));

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(a[i], b[i]);
    }
}
