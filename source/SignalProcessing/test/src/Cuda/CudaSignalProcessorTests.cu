#include <SignalProcessing/Cuda/CudaSignalProcessor.h>

#include <gtest/gtest.h>

#include <chrono>
#include <iostream>

using namespace adaptone;
using namespace std;

TEST(CudaSignalProcessorTests, process_shouldCopyTheInputToTheOutput)
{
    constexpr size_t frameSampleCount = 2;
    constexpr size_t sampleFrequency = 48000;
    constexpr size_t inputChannelCount = 1;
    constexpr size_t outputChannelCount = 1;
    constexpr PcmAudioFrame::Format inputFormat = PcmAudioFrame::Format::Signed8;
    constexpr PcmAudioFrame::Format outputFormat = PcmAudioFrame::Format::Signed8;
    constexpr size_t parametricEqFilterCount = 2;
    vector<double> frequencies{ 20, 50, 125 };

    CudaSignalProcessor<float> processor(frameSampleCount,
        sampleFrequency,
        inputChannelCount,
        outputChannelCount,
        inputFormat,
        outputFormat,
        parametricEqFilterCount,
        frequencies);

    PcmAudioFrame inputFrame(outputFormat, outputChannelCount, frameSampleCount);

    inputFrame[0] = 5;
    inputFrame[1] = 10;
    PcmAudioFrame outputFrame = processor.process(inputFrame);

    EXPECT_EQ(outputFrame[0], 5);
    EXPECT_EQ(outputFrame[1], 10);

    inputFrame[0] = 15;
    inputFrame[1] = 25;
    outputFrame = processor.process(inputFrame);
    EXPECT_EQ(outputFrame[0], 15);
    EXPECT_EQ(outputFrame[1], 25);
}

TEST(CudaSignalProcessorTests, performance)
{
    constexpr size_t frameSampleCount = 32;
    constexpr size_t sampleFrequency = 48000;
    constexpr size_t inputChannelCount = 16;
    constexpr size_t outputChannelCount = 16;
    constexpr PcmAudioFrame::Format inputFormat = PcmAudioFrame::Format::SignedPadded24;
    constexpr PcmAudioFrame::Format outputFormat = PcmAudioFrame::Format::SignedPadded24;
    constexpr size_t parametricEqFilterCount = 2;
    vector<double> frequencies{ 20, 50, 125 };

    CudaSignalProcessor<float> processor(frameSampleCount,
        sampleFrequency,
        inputChannelCount,
        outputChannelCount,
        inputFormat,
        outputFormat,
        parametricEqFilterCount,
        frequencies);

    PcmAudioFrame inputFrame(outputFormat, outputChannelCount, frameSampleCount);


    constexpr std::size_t Count = 10000;

    auto start = chrono::system_clock::now();
    for (std::size_t i = 0; i < Count; i++)
    {
        for (std::size_t j = 0; j < inputFrame.size(); j++)
        {
            inputFrame[j] = 10;
        }
        PcmAudioFrame outputFrame = processor.process(inputFrame);
    }
    auto end = chrono::system_clock::now();
    chrono::duration<double> elapsedSeconds = end - start;

    cout << "Elapsed time = " << elapsedSeconds.count() / Count << " s" << endl;
}

//TODO Faire un test où les paramètres changent un après l'autre.
