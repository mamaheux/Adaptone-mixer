#include <SignalProcessing/Cuda/CudaSignalProcessor.h>

#include <gtest/gtest.h>

#include <chrono>
#include <iostream>

using namespace adaptone;
using namespace std;

TEST(CudaSignalProcessorTests, performance)
{
    constexpr size_t frameSampleCount = 32;
    constexpr size_t sampleFrequency = 48000;
    constexpr size_t inputChannelCount = 16;
    constexpr size_t outputChannelCount = 16;
    constexpr PcmAudioFrame::Format inputFormat = PcmAudioFrame::Format::SignedPadded24;
    constexpr PcmAudioFrame::Format outputFormat = PcmAudioFrame::Format::SignedPadded24;
    vector<double> frequencies{ 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
        1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000 };
    constexpr size_t soundLevelLength = 2;
    shared_ptr<AnalysisDispatcher> analysisDispatcher;

    CudaSignalProcessor<float> processor(frameSampleCount,
        sampleFrequency,
        inputChannelCount,
        outputChannelCount,
        inputFormat,
        outputFormat,
        frequencies,
        soundLevelLength,
        analysisDispatcher);

    PcmAudioFrame inputFrame(outputFormat, outputChannelCount, frameSampleCount);


    constexpr size_t Count = 10000;

    auto start = chrono::system_clock::now();
    for (size_t i = 0; i < Count; i++)
    {
        for (size_t j = 0; j < inputFrame.size(); j++)
        {
            inputFrame[j] = 10;
        }
        PcmAudioFrame outputFrame = processor.process(inputFrame);
    }
    auto end = chrono::system_clock::now();
    chrono::duration<double> elapsedSeconds = end - start;

    cout << "Elapsed time = " << elapsedSeconds.count() / Count << " s" << endl;
}

TEST(CudaSignalProcessorTests, process_shouldConsiderVariableParameters)
{
    constexpr size_t frameSampleCount = 2;
    constexpr size_t sampleFrequency = 48000;
    constexpr size_t inputChannelCount = 2;
    constexpr size_t outputChannelCount = 2;
    constexpr PcmAudioFrame::Format inputFormat = PcmAudioFrame::Format::Signed8;
    constexpr PcmAudioFrame::Format outputFormat = PcmAudioFrame::Format::Signed8;
    vector<double> frequencies{ 20, 50, 125 };
    constexpr size_t soundLevelLength = 2;
    shared_ptr<AnalysisDispatcher> analysisDispatcher;

    CudaSignalProcessor<double> processor(frameSampleCount,
        sampleFrequency,
        inputChannelCount,
        outputChannelCount,
        inputFormat,
        outputFormat,
        frequencies,
        soundLevelLength,
        analysisDispatcher);

    vector<double> inputGains = {1, 1};
    vector<double> mixingGains = {1, 0, 0, 1};
    vector<double> outputGains = {1, 1};

    processor.setInputGains(inputGains);
    processor.setMixingGains(mixingGains);
    processor.setOutputGains(outputGains);

    processor.setInputGraphicEqGains(0, {0, 0, 0});
    processor.setInputGraphicEqGains(1, {0, 0, 0});

    processor.forceRefreshParameters();

    PcmAudioFrame inputFrame(outputFormat, outputChannelCount, frameSampleCount);
    inputFrame[0] = 5;
    inputFrame[1] = 10;
    inputFrame[2] = 15;
    inputFrame[3] = 20;

    PcmAudioFrame outputFrame = processor.process(inputFrame);

    EXPECT_EQ(outputFrame[0], 5);
    EXPECT_EQ(outputFrame[1], 10);

    vector<double> newInputGains = {2, 2};
    processor.setInputGains(newInputGains);

    outputFrame = processor.process(inputFrame);

    EXPECT_EQ(outputFrame[0], 5);
    EXPECT_EQ(outputFrame[1], 10);

    vector<double> newMixingGains = {2, 2, 2, 2};
    processor.setMixingGains(newMixingGains);

    outputFrame = processor.process(inputFrame);

    EXPECT_EQ(outputFrame[0], 5);
    EXPECT_EQ(outputFrame[1], 10);

    vector<double> newOutputGains = {2, 2};
    processor.setOutputGains(newOutputGains);

    outputFrame = processor.process(inputFrame);

    EXPECT_EQ(outputFrame[0], 5);
    EXPECT_EQ(outputFrame[1], 10);
}
