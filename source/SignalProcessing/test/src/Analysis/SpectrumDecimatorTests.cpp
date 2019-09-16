#include <SignalProcessing/Analysis/SpectrumDecimator.h>

#include <Utils/Exception/InvalidValueException.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

constexpr double MaxAbsFrequencyError = 0.1;

TEST(SpectrumDecimatorTests, buckets_shouldReturnTheRightBucketsForTheSpecifiedParameters)
{
    constexpr size_t SpectrumSize = 2048;
    constexpr size_t SampleFrequency = 48000;
    constexpr size_t PointCountPerDecade = 2;

    SpectrumDecimator spectrumDecimator(SpectrumSize, SampleFrequency, PointCountPerDecade);

    EXPECT_EQ(spectrumDecimator.buckets().size(), 8);

    EXPECT_NEAR(spectrumDecimator.buckets()[0].frequency(), 5.85938, MaxAbsFrequencyError);
    EXPECT_EQ(spectrumDecimator.buckets()[0].lowerBoundIndex(), 0);
    EXPECT_EQ(spectrumDecimator.buckets()[0].upperBoundIndex(), 1);

    EXPECT_NEAR(spectrumDecimator.buckets()[1].frequency(), 41.0156, MaxAbsFrequencyError);
    EXPECT_EQ(spectrumDecimator.buckets()[1].lowerBoundIndex(), 2);
    EXPECT_EQ(spectrumDecimator.buckets()[1].upperBoundIndex(), 5);

    EXPECT_NEAR(spectrumDecimator.buckets()[2].frequency(), 128.906, MaxAbsFrequencyError);
    EXPECT_EQ(spectrumDecimator.buckets()[2].lowerBoundIndex(), 6);
    EXPECT_EQ(spectrumDecimator.buckets()[2].upperBoundIndex(), 16);

    EXPECT_NEAR(spectrumDecimator.buckets()[3].frequency(), 357.422, MaxAbsFrequencyError);
    EXPECT_EQ(spectrumDecimator.buckets()[3].lowerBoundIndex(), 17);
    EXPECT_EQ(spectrumDecimator.buckets()[3].upperBoundIndex(), 44);

    EXPECT_NEAR(spectrumDecimator.buckets()[4].frequency(), 943.359, MaxAbsFrequencyError);
    EXPECT_EQ(spectrumDecimator.buckets()[4].lowerBoundIndex(), 45);
    EXPECT_EQ(spectrumDecimator.buckets()[4].upperBoundIndex(), 116);

    EXPECT_NEAR(spectrumDecimator.buckets()[5].frequency(), 2460.94, MaxAbsFrequencyError);
    EXPECT_EQ(spectrumDecimator.buckets()[5].lowerBoundIndex(), 117);
    EXPECT_EQ(spectrumDecimator.buckets()[5].upperBoundIndex(), 303);

    EXPECT_NEAR(spectrumDecimator.buckets()[6].frequency(), 6398.44, MaxAbsFrequencyError);
    EXPECT_EQ(spectrumDecimator.buckets()[6].lowerBoundIndex(), 304);
    EXPECT_EQ(spectrumDecimator.buckets()[6].upperBoundIndex(), 788);

    EXPECT_NEAR(spectrumDecimator.buckets()[7].frequency(), 16617.2, MaxAbsFrequencyError);
    EXPECT_EQ(spectrumDecimator.buckets()[7].lowerBoundIndex(), 789);
    EXPECT_EQ(spectrumDecimator.buckets()[7].upperBoundIndex(), 2047);
}

TEST(SpectrumDecimatorTests, getDecimatedAmplitudes_invalidSize_shouldThrowInvalidValueException)
{
    constexpr size_t SpectrumSize = 2048;
    constexpr size_t SampleFrequency = 48000;
    constexpr size_t PointCountPerDecade = 2;

    SpectrumDecimator spectrumDecimator(SpectrumSize, SampleFrequency, PointCountPerDecade);

    arma::fvec positiveFrequencyAmplitudes = arma::ones<arma::fvec>(2047);
    EXPECT_THROW(spectrumDecimator.getDecimatedAmplitudes(positiveFrequencyAmplitudes), InvalidValueException);
}

TEST(SpectrumDecimatorTests, getDecimatedAmplitudes_shouldReturnTheDecimatedAmplitudes)
{
    constexpr size_t SpectrumSize = 2048;
    constexpr size_t SampleFrequency = 48000;
    constexpr size_t PointCountPerDecade = 2;

    SpectrumDecimator spectrumDecimator(SpectrumSize, SampleFrequency, PointCountPerDecade);

    arma::fvec positiveFrequencyAmplitudes = arma::ones<arma::fvec>(2);
    positiveFrequencyAmplitudes = arma::join_vert(positiveFrequencyAmplitudes, 5 * arma::ones<arma::fvec>(4));
    positiveFrequencyAmplitudes = arma::join_vert(positiveFrequencyAmplitudes, 10 * arma::ones<arma::fvec>(11));
    positiveFrequencyAmplitudes = arma::join_vert(positiveFrequencyAmplitudes, 50 * arma::ones<arma::fvec>(28));
    positiveFrequencyAmplitudes = arma::join_vert(positiveFrequencyAmplitudes, 100 * arma::ones<arma::fvec>(72));
    positiveFrequencyAmplitudes = arma::join_vert(positiveFrequencyAmplitudes, 500 * arma::ones<arma::fvec>(187));
    positiveFrequencyAmplitudes = arma::join_vert(positiveFrequencyAmplitudes, 1000 * arma::ones<arma::fvec>(485));
    positiveFrequencyAmplitudes = arma::join_vert(positiveFrequencyAmplitudes, 5000 * arma::ones<arma::fvec>(1259));

    vector<SpectrumPoint> decimatedAmplitudes =
        spectrumDecimator.getDecimatedAmplitudes(positiveFrequencyAmplitudes);

    EXPECT_EQ(decimatedAmplitudes.size(), 8);

    EXPECT_NEAR(decimatedAmplitudes[0].frequency(), 5.85938, MaxAbsFrequencyError);
    EXPECT_DOUBLE_EQ(decimatedAmplitudes[0].amplitude(), 1);

    EXPECT_NEAR(decimatedAmplitudes[1].frequency(), 41.0156, MaxAbsFrequencyError);
    EXPECT_DOUBLE_EQ(decimatedAmplitudes[1].amplitude(), 5);

    EXPECT_NEAR(decimatedAmplitudes[2].frequency(), 128.906, MaxAbsFrequencyError);
    EXPECT_DOUBLE_EQ(decimatedAmplitudes[2].amplitude(), 10);

    EXPECT_NEAR(decimatedAmplitudes[3].frequency(), 357.422, MaxAbsFrequencyError);
    EXPECT_DOUBLE_EQ(decimatedAmplitudes[3].amplitude(), 50);

    EXPECT_NEAR(decimatedAmplitudes[4].frequency(), 943.359, MaxAbsFrequencyError);
    EXPECT_DOUBLE_EQ(decimatedAmplitudes[4].amplitude(), 100);

    EXPECT_NEAR(decimatedAmplitudes[5].frequency(), 2460.94, MaxAbsFrequencyError);
    EXPECT_DOUBLE_EQ(decimatedAmplitudes[5].amplitude(), 500);

    EXPECT_NEAR(decimatedAmplitudes[6].frequency(), 6398.44, MaxAbsFrequencyError);
    EXPECT_DOUBLE_EQ(decimatedAmplitudes[6].amplitude(), 1000);

    EXPECT_NEAR(decimatedAmplitudes[7].frequency(), 16617.2, MaxAbsFrequencyError);
    EXPECT_DOUBLE_EQ(decimatedAmplitudes[7].amplitude(), 5000);
}

TEST(SpectrumDecimatorTests, getDecimatedAmplitudes_performance)
{
    constexpr size_t IterationCount = 10000;

    constexpr size_t SpectrumSize = 2048;
    constexpr size_t SampleFrequency = 48000;
    constexpr size_t PointCountPerDecade = 10;
    SpectrumDecimator spectrumDecimator(SpectrumSize, SampleFrequency, PointCountPerDecade);

    auto start = chrono::system_clock::now();
    arma::fvec positiveFrequencyAmplitudes = arma::ones<arma::fvec>(SpectrumSize);
    for (size_t i = 0; i < IterationCount; i++)
    {
        vector<SpectrumPoint> decimatedAmplitudes =
            spectrumDecimator.getDecimatedAmplitudes(positiveFrequencyAmplitudes);
    }

    auto end = chrono::system_clock::now();
    chrono::duration<double> elapsedSeconds = end - start;
    cout << "Elapsed time (avg) = " << elapsedSeconds.count() / IterationCount << " s" << endl;
}
