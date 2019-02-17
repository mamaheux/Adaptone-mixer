#include <SignalProcessing/Cuda/CudaSignalProcessor.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(CudaSignalProcessorTests, process_shouldCopyTheInputToTheOutput)
{
    size_t frameSampleCount = 2;
    size_t sampleFrequency = 48000;
    size_t inputChannelCount = 1;
    size_t outputChannelCount = 1;
    PcmAudioFrame::Format inputFormat = PcmAudioFrame::Format::Signed8;
    PcmAudioFrame::Format outputFormat = PcmAudioFrame::Format::Signed8;

    CudaSignalProcessor<float> processor(frameSampleCount,
        sampleFrequency,
        inputChannelCount,
        outputChannelCount,
        inputFormat,
        outputFormat);

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
