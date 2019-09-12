#include <SignalProcessing/Analysis/RealtimeSpectrumAnalyser.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(RealtimeSpectrumAnalyserTests, calculateFftAnalysis_shouldReturnTheFftOfEachChannel)
{
    constexpr size_t FftSize = 4;
    constexpr size_t SampleFrequency = 48000;
    constexpr size_t ChannelCount = 2;
    constexpr size_t DecimatorPointCountPerDecade = 2;

    RealtimeSpectrumAnalyser analyser(FftSize, SampleFrequency, ChannelCount, DecimatorPointCountPerDecade);

    analyser.writePartialData([](size_t i, float* b)
    {
        b[FftSize * i] = 0;
        b[FftSize * i + 1] = 1;
        b[FftSize * i + 2] = 2;
        b[FftSize * i + 3] = 3;
    });

    analyser.writePartialData([](size_t i, float* b)
    {
        b[FftSize * i] = 1;
        b[FftSize * i + 1] = 1;
        b[FftSize * i + 2] = 2;
        b[FftSize * i + 3] = 2;
    });

    analyser.finishWriting();
    vector<arma::cx_fvec> channelFfts = analyser.calculateFftAnalysis();

    ASSERT_EQ(channelFfts.size(), ChannelCount);

    ASSERT_EQ(channelFfts[0].n_elem, 3);
    EXPECT_FLOAT_EQ(channelFfts[0](0).real(), 2.55);
    EXPECT_FLOAT_EQ(channelFfts[0](0).imag(), 0);
    EXPECT_FLOAT_EQ(channelFfts[0](1).real(), -1.54);
    EXPECT_FLOAT_EQ(channelFfts[0](1).imag(), -0.52999997);
    EXPECT_FLOAT_EQ(channelFfts[0](2).real(), 0.53);
    EXPECT_FLOAT_EQ(channelFfts[0](2).imag(), 0);

    ASSERT_EQ(channelFfts[1].n_elem, 3);
    EXPECT_FLOAT_EQ(channelFfts[1](0).real(), 2.55);
    EXPECT_FLOAT_EQ(channelFfts[1](0).imag(), 0);
    EXPECT_FLOAT_EQ(channelFfts[1](1).real(), -1.46);
    EXPECT_FLOAT_EQ(channelFfts[1](1).imag(), -0.61000001);
    EXPECT_FLOAT_EQ(channelFfts[1](2).real(), 0.69);
    EXPECT_FLOAT_EQ(channelFfts[1](2).imag(), 0);
}

TEST(RealtimeSpectrumAnalyserTests, calculateDecimatedSpectrumAnalysis_shouldReturnTheDecimatedAnalysisForEachChannel)
{
    constexpr size_t FftSize = 4;
    constexpr size_t SampleFrequency = 48000;
    constexpr size_t ChannelCount = 2;
    constexpr size_t DecimatorPointCountPerDecade = 2;

    RealtimeSpectrumAnalyser analyser(FftSize, SampleFrequency, ChannelCount, DecimatorPointCountPerDecade);

    analyser.writePartialData([](size_t i, float* b)
    {
        b[FftSize * i] = 0;
        b[FftSize * i + 1] = 1;
        b[FftSize * i + 2] = 2;
        b[FftSize * i + 3] = 3;
    });

    analyser.writePartialData([](size_t i, float* b)
    {
        b[FftSize * i] = 1;
        b[FftSize * i + 1] = 1;
        b[FftSize * i + 2] = 2;
        b[FftSize * i + 3] = 2;
    });

    analyser.finishWriting();
    vector<vector<SpectrumPoint>> result = analyser.calculateDecimatedSpectrumAnalysis();

    ASSERT_EQ(result.size(), ChannelCount);

    ASSERT_EQ(result[0].size(), 2);
    EXPECT_DOUBLE_EQ(result[0][0].frequency(), 8000);
    EXPECT_DOUBLE_EQ(result[0][0].amplitude(), 1.6286497116088867);
    EXPECT_DOUBLE_EQ(result[0][1].frequency(), 16000);
    EXPECT_DOUBLE_EQ(result[0][1].amplitude(), 0.52999985218048096);

    ASSERT_EQ(result[1].size(), 2);
    EXPECT_DOUBLE_EQ(result[1][0].frequency(), 8000);
    EXPECT_DOUBLE_EQ(result[1][0].amplitude(), 1.5823084115982056);
    EXPECT_DOUBLE_EQ(result[1][1].frequency(), 16000);
    EXPECT_DOUBLE_EQ(result[1][1].amplitude(), 0.68999993801116943);
}


TEST(RealtimeSpectrumAnalyserTests, calculateDecimatedSpectrumAnalysis_performance)
{
    constexpr size_t IterationCount = 1000;

    constexpr size_t FftSize = 4096;
    constexpr size_t SampleFrequency = 48000;
    constexpr size_t ChannelCount = 16;
    constexpr size_t DecimatorPointCountPerDecade = 10;

    RealtimeSpectrumAnalyser analyser(FftSize, SampleFrequency, ChannelCount, DecimatorPointCountPerDecade);

    auto start = chrono::system_clock::now();
    for (size_t i = 0; i < IterationCount; i++)
    {
        analyser.finishWriting();
        vector<vector<SpectrumPoint>> result = analyser.calculateDecimatedSpectrumAnalysis();
    }

    auto end = chrono::system_clock::now();
    chrono::duration<double> elapsedSeconds = end - start;
    cout << "Elapsed time (avg) = " << elapsedSeconds.count() / IterationCount << " s" << endl;
}
