#include <SignalProcessing/Cuda/CudaSignalProcessor.h>

#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <limits>

using namespace adaptone;
using namespace std;

TEST(CudaSignalProcessorTests, performance)
{
    constexpr size_t FrameSampleCount = 32;
    constexpr size_t SampleFrequency = 48000;
    constexpr size_t InputChannelCount = 16;
    constexpr size_t OutputChannelCount = 16;
    constexpr PcmAudioFrameFormat InputFormat = PcmAudioFrameFormat::SignedPadded24;
    constexpr PcmAudioFrameFormat OutputFormat = PcmAudioFrameFormat::SignedPadded24;
    vector<double> frequencies{ 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
        1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000 };
    constexpr std::size_t MaxOutputDelay = 441088;
    constexpr size_t SoundLevelLength = 4096;
    shared_ptr<AnalysisDispatcher> analysisDispatcher;

    CudaSignalProcessor<float> processor(analysisDispatcher, SignalProcessorParameters(ProcessingDataType::Float,
        FrameSampleCount,
        SampleFrequency,
        InputChannelCount,
        OutputChannelCount,
        InputFormat,
        OutputFormat,
        frequencies,
        MaxOutputDelay,
        SoundLevelLength));

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
    constexpr size_t InputChannelCount = 3;
    constexpr size_t OutputChannelCount = 2;
    constexpr PcmAudioFrameFormat InputFormat = PcmAudioFrameFormat::Double;
    constexpr PcmAudioFrameFormat OutputFormat = PcmAudioFrameFormat::Double;
    vector<double> frequencies{ 20, 10000, 20000 };
    constexpr std::size_t MaxOutputDelay = 4 * FrameSampleCount;
    constexpr size_t SoundLevelLength = 4096;
    shared_ptr<AnalysisDispatcher> analysisDispatcher;

    CudaSignalProcessor<double> processor(analysisDispatcher, SignalProcessorParameters(ProcessingDataType::Double,
        FrameSampleCount,
        SampleFrequency,
        InputChannelCount,
        OutputChannelCount,
        InputFormat,
        OutputFormat,
        frequencies,
        MaxOutputDelay,
        SoundLevelLength));

    processor.setInputGains({ 1, 1, 0 });
    processor.setMixingGains({ 1, 0, 0, 0, 1, 0 });
    processor.setOutputGains({ 1, 1 });

    processor.forceRefreshParameters();

    PcmAudioFrame inputFrame(InputFormat, InputChannelCount, FrameSampleCount);
    double* inputFrameDouble = reinterpret_cast<double*>(&inputFrame[0]);
    inputFrameDouble[0] = 0.005;
    inputFrameDouble[1] = 0.01;
    inputFrameDouble[2] = 0;
    inputFrameDouble[3] = 0.015;
    inputFrameDouble[4] = 0.02;
    inputFrameDouble[5] = 0;


    const double* outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.005, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.01, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.015, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.02, MaxAbsError);


    processor.setInputGains({ 2, 2, 0 });
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.01, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.02, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.03, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.04, MaxAbsError);


    processor.setInputGraphicEqGains(0, { 2, 2, 2 });
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.02, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.02, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.06, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.04, MaxAbsError);


    processor.setInputGraphicEqGains(1, { 2, 2, 2 });
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.02, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.04, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.06, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.08, MaxAbsError);


    processor.setMixingGains({ 0, 1, 0, 1, 0, 0 });
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.04, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.02, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.08, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.06, MaxAbsError);


    processor.setOutputGraphicEqGains(0, { 2, 2, 2 });
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.08, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.02, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.16, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.06, MaxAbsError);


    processor.setOutputGraphicEqGains(1, { 2, 2, 2 });
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.08, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.04, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.16, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.12, MaxAbsError);


    processor.setUniformizationGraphicEqGains(0, { 2, 2, 2 });
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.16, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.04, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.32, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.12, MaxAbsError);


    processor.setUniformizationGraphicEqGains(1, { 2, 2, 2 });
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.16, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.08, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.32, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.24, MaxAbsError);

    processor.setUniformizationGraphicEqGains(0, { 1, 1, 1 });
    processor.setUniformizationGraphicEqGains(1, { 1, 1, 1 });
    processor.setOutputGains({ 2, 2 });
    processor.process(inputFrame);
    processor.process(inputFrame);
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.16, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.08, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.32, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.24, MaxAbsError);


    processor.setInputGain(0, 4);
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.16, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.16, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.32, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.48, MaxAbsError);


    processor.setInputGain(1, 4);
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.32, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.16, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.64, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.48, MaxAbsError);


    processor.setOutputGain(0, 1);
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.16, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.16, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.32, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.48, MaxAbsError);


    processor.setOutputGain(1, 1);
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.16, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.08, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.32, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.24, MaxAbsError);


    processor.setOutputGraphicEqGains(0, 2, { 1, 1, 1 });
    processor.process(inputFrame);
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0.08, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.04, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0.16, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.12, MaxAbsError);


    processor.setOutputDelay(0, 1);
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[2], 0.08, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.04, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.12, MaxAbsError);


    processor.setOutputDelay(1, 1);
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[2], 0.08, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.04, MaxAbsError);


    processor.setOutputDelays({0, 0});
    processor.setMixingGain(1, 0, 0);
    processor.process(inputFrame);
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0.04, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0.12, MaxAbsError);


    processor.setMixingGains(1, { 0, 0, 0 });
    outputFrameDouble = reinterpret_cast<const double*>(processor.process(inputFrame).data());

    EXPECT_NEAR(outputFrameDouble[0], 0, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[1], 0, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[2], 0, MaxAbsError);
    EXPECT_NEAR(outputFrameDouble[3], 0, MaxAbsError);
}
