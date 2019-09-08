#include <SignalProcessing/Analysis/Math.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(MathTests, hamming_shouldReturnTheHammingWindow)
{
    constexpr size_t WindowLength = 5;
    arma::fvec window = hamming<arma::fvec>(WindowLength);

    EXPECT_EQ(window.n_elem, WindowLength);
    EXPECT_FLOAT_EQ(window(0), 0.08);
    EXPECT_FLOAT_EQ(window(1), 0.54);
    EXPECT_FLOAT_EQ(window(2), 1);
    EXPECT_FLOAT_EQ(window(3), 0.540000);
    EXPECT_FLOAT_EQ(window(4), 0.08);
}
