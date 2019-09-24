#include <SignalProcessing/Utils/Math.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

static constexpr double MaxAbsError = 0.0001;

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

TEST(MathTests, fft_preallocatedMemory_shouldReturnTheFft)
{
    arma::cx_vec x = arma::conv_to<arma::cx_vec>::from(arma::vec({ 2, 2, 2, 2 }));
    arma::cx_vec y = arma::zeros<arma::cx_vec>(x.n_elem);

    fft(x, y);

    EXPECT_EQ(y(0), complex<double>(8, 0));
    EXPECT_EQ(y(1), complex<double>(0, 0));
    EXPECT_EQ(y(2), complex<double>(0, 0));
    EXPECT_EQ(y(3), complex<double>(0, 0));
}

TEST(MathTests, fft_shouldReturnTheFft)
{
    arma::cx_vec x = arma::conv_to<arma::cx_vec>::from(arma::vec({ 1, 1, 1 }));
    arma::cx_vec y;

    fft(x, y);

    EXPECT_EQ(y(0), complex<double>(3, 0));
    EXPECT_EQ(y(1), complex<double>(0, 0));
    EXPECT_EQ(y(2), complex<double>(0, 0));
}

TEST(MathTests, ifft_preallocatedMemory_shouldReturnTheIfft)
{
    arma::cx_vec x({ complex<double>(8, 0), complex<double>(0, 0), complex<double>(0, 0), complex<double>(0, 0) });
    arma::cx_vec y = arma::zeros<arma::cx_vec>(x.n_elem);

    ifft(x, y);

    EXPECT_EQ(y(0), complex<double>(2, 0));
    EXPECT_EQ(y(1), complex<double>(2, 0));
    EXPECT_EQ(y(2), complex<double>(2, 0));
    EXPECT_EQ(y(3), complex<double>(2, 0));
}

TEST(MathTests, ifft_shouldReturnTheIfft)
{
    arma::cx_vec x({ complex<double>(3, 0), complex<double>(0, 0), complex<double>(0, 0) });
    arma::cx_vec y;

    ifft(x, y);

    EXPECT_EQ(y(0), complex<double>(1, 0));
    EXPECT_EQ(y(1), complex<double>(1, 0));
    EXPECT_EQ(y(2), complex<double>(1, 0));
}

TEST(MathTests, hilbert_evenSize_shouldReturnTheAnalyticSignal)
{
    arma::cx_vec x = arma::conv_to<arma::cx_vec>::from(arma::vec({ 0, 0, 2, 2 }));
    arma::cx_vec y;

    hilbert(x, y);

    EXPECT_NEAR(y(0).real(), 0, MaxAbsError);
    EXPECT_NEAR(y(0).imag(), 1, MaxAbsError);

    EXPECT_NEAR(y(1).real(), 0, MaxAbsError);
    EXPECT_NEAR(y(1).imag(), -1, MaxAbsError);

    EXPECT_NEAR(y(2).real(), 2, MaxAbsError);
    EXPECT_NEAR(y(2).imag(), -1, MaxAbsError);

    EXPECT_NEAR(y(3).real(), 2, MaxAbsError);
    EXPECT_NEAR(y(3).imag(), 1, MaxAbsError);
}

TEST(MathTests, hilbert_oddSize_shouldReturnTheAnalyticSignal)
{
    arma::cx_vec x = arma::conv_to<arma::cx_vec>::from(arma::vec({ 1, 2, 3, 4, 5 }));
    arma::cx_vec y;

    hilbert(x, y);

    EXPECT_NEAR(y(0).real(), 1, MaxAbsError);
    EXPECT_NEAR(y(0).imag(), 1.7013, MaxAbsError);

    EXPECT_NEAR(y(1).real(), 2, MaxAbsError);
    EXPECT_NEAR(y(1).imag(), -1.3763, MaxAbsError);

    EXPECT_NEAR(y(2).real(), 3, MaxAbsError);
    EXPECT_NEAR(y(2).imag(), -0.6498, MaxAbsError);

    EXPECT_NEAR(y(3).real(), 4, MaxAbsError);
    EXPECT_NEAR(y(3).imag(), -1.3763, MaxAbsError);

    EXPECT_NEAR(y(4).real(), 5, MaxAbsError);
    EXPECT_NEAR(y(4).imag(), 1.7013, MaxAbsError);
}

TEST(MathTests, interpolateWithNaNRemoval_shouldRemoveNaNValues)
{
    arma::vec x({ 1, 2, 3 });
    arma::vec y({ 1, 2, 3 });

    arma::vec xx({ 0, 1, 2, 3, 4, });
    arma::vec yy;

    interpolateWithNaNRemoval(x, y, xx, yy);
    EXPECT_NEAR(yy(0), 1, MaxAbsError);
    EXPECT_NEAR(yy(1), 1, MaxAbsError);
    EXPECT_NEAR(yy(2), 2, MaxAbsError);
    EXPECT_NEAR(yy(3), 3, MaxAbsError);
    EXPECT_NEAR(yy(4), 3, MaxAbsError);
}
