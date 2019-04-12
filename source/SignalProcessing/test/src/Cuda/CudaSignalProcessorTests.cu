#include <SignalProcessing/Cuda/CudaSignalProcessor.h>

#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <limits>

using namespace adaptone;
using namespace std;

constexpr double NInf = -numeric_limits<double>::infinity();

TEST(CudaSignalProcessorTests, performance)
{
    constexpr size_t FrameSampleCount = 32;
    constexpr size_t SampleFrequency = 48000;
    constexpr size_t InputChannelCount = 16;
    constexpr size_t OutputChannelCount = 16;
    constexpr PcmAudioFrame::Format InputFormat = PcmAudioFrame::Format::SignedPadded24;
    constexpr PcmAudioFrame::Format OutputFormat = PcmAudioFrame::Format::SignedPadded24;
    vector<double> frequencies{ 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
        1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000 };
    constexpr size_t SoundLevelLength = 4096;
    shared_ptr<AnalysisDispatcher> analysisDispatcher;

    CudaSignalProcessor<float> processor(FrameSampleCount,
        SampleFrequency,
        InputChannelCount,
        OutputChannelCount,
        InputFormat,
        OutputFormat,
        frequencies,
        SoundLevelLength,
        analysisDispatcher);

    PcmAudioFrame inputFrame(OutputFormat, OutputChannelCount, FrameSampleCount);


    constexpr size_t Count = 10000;

    double minElapsedTimeSeconds = 1;
    double maxElapsedTimeSeconds = 0;
    double totalElapsedTimeSeconds = 0;

    for (size_t i = 0; i < Count; i++)
    {
        auto start = chrono::system_clock::now();

        for (size_t j = 0; j < inputFrame.size(); j++)
        {
            inputFrame[j] = 10;
        }
        PcmAudioFrame outputFrame = processor.process(inputFrame);

        auto end = chrono::system_clock::now();
        chrono::duration<double> elapsedSeconds = end - start;

        totalElapsedTimeSeconds += elapsedSeconds.count();
        minElapsedTimeSeconds = elapsedSeconds.count() < minElapsedTimeSeconds ? elapsedSeconds.count() : minElapsedTimeSeconds;
        maxElapsedTimeSeconds = elapsedSeconds.count() > maxElapsedTimeSeconds ? elapsedSeconds.count() : maxElapsedTimeSeconds;
    }

    cout << "Elapsed time (avg) = " << totalElapsedTimeSeconds / Count << " s" << endl;
    cout << "Elapsed time (min) = " << minElapsedTimeSeconds << " s" << endl;
    cout << "Elapsed time (max) = " << maxElapsedTimeSeconds << " s" << endl;
}

TEST(CudaSignalProcessorTests, process_shouldConsiderVariableParameters)
{
    constexpr double MaxAbsError = 0.00001;

    constexpr size_t FrameSampleCount = 2;
    constexpr size_t SampleFrequency = 48000;
    constexpr size_t InputChannelCount = 2;
    constexpr size_t OutputChannelCount = 2;
    constexpr PcmAudioFrame::Format InputFormat = PcmAudioFrame::Format::Double;
    constexpr PcmAudioFrame::Format OutputFormat = PcmAudioFrame::Format::Double;
    vector<double> frequencies{ 20, 10000, 20000 };
    constexpr size_t SoundLevelLength = 4096;
    shared_ptr<AnalysisDispatcher> analysisDispatcher;

    CudaSignalProcessor<double> processor(FrameSampleCount,
        SampleFrequency,
        InputChannelCount,
        OutputChannelCount,
        InputFormat,
        OutputFormat,
        frequencies,
        SoundLevelLength,
        analysisDispatcher);

    processor.setInputGains({ 0, 0 });
    processor.setMixingGains({ 0, NInf, NInf, 0 });
    processor.setOutputGains({ 0, 0 });

    processor.setInputGraphicEqGains(0, { 0, 0, 0 });
    processor.setInputGraphicEqGains(1, { 0, 0, 0 });
    processor.setOutputGraphicEqGains(0, { 0, 0, 0 });
    processor.setOutputGraphicEqGains(1, { 0, 0, 0 });

    processor.forceRefreshParameters();

    PcmAudioFrame inputFrame(OutputFormat, OutputChannelCount, FrameSampleCount);
    double* inputFrameDouble = reinterpret_cast<double*>(&inputFrame[0]);
    inputFrameDouble[0] = 0.005;
    inputFrameDouble[1] = 0.01;
    inputFrameDouble[2] = 0.015;
    inputFrameDouble[3] = 0.02;


    const double* outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.005, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.01, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.015, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.02, MaxAbsError);


    processor.setInputGains({ 6.02059991328, 6.02059991328 });
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.01, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.02, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.03, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.04, MaxAbsError);


    processor.setInputGraphicEqGains(0, { 6.02059991328, 6.02059991328, 6.02059991328 });
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.02, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.02, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.06, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.04, MaxAbsError);


    processor.setInputGraphicEqGains(1, { 6.02059991328, 6.02059991328, 6.02059991328 });
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.02, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.04, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.06, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.08, MaxAbsError);


    processor.setMixingGains({ NInf, 0, 0, NInf });
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.04, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.02, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.08, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.06, MaxAbsError);


    processor.setOutputGraphicEqGains(0, { 6.02059991328, 6.02059991328, 6.02059991328 });
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.08, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.02, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.16, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.06, MaxAbsError);


    processor.setOutputGraphicEqGains(1, { 6.02059991328, 6.02059991328, 6.02059991328 });
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.08, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.04, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.16, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.12, MaxAbsError);


    processor.setOutputGains({ 6.02059991328, 6.02059991328 });
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.16, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.08, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.32, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.24, MaxAbsError);


    processor.setInputGain(0, 12.0411998266);
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.16, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.16, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.32, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.48, MaxAbsError);


    processor.setInputGain(1, 12.0411998266);
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.32, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.16, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.64, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.48, MaxAbsError);


    processor.setOutputGain(0, 0);
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.16, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.16, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.32, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.48, MaxAbsError);


    processor.setOutputGain(1, 0);
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.16, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.08, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.32, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.24, MaxAbsError);


    processor.setOutputGraphicEqGains(0, 2, { 0, 0, 0 });
    reinterpret_cast<const double*>(processor.process(inputFrame).data());
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.08, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.04, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.16, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.12, MaxAbsError);


    processor.setMixingGain(1, 0, NInf);
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.04, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.12, MaxAbsError);


    processor.setMixingGains(1, { NInf, NInf });
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0, MaxAbsError);
}
