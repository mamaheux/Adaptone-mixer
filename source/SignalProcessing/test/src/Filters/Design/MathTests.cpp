#include <SignalProcessing/Filters/Design/Math.h>

#include <gtest/gtest.h>

#include <iostream>

using namespace adaptone;

static constexpr double MaxAbsError = 0.0001;

TEST(MathTests, hilbert_evenSize_shouldReturnTheAnalyticSignal)
{
    arma::vec x({ 0, 0, 2, 2 });
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
    arma::vec x({ 1, 2, 3, 4, 5 });
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
