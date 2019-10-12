#include <SignalProcessing/SignalProcessorParameters.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(SignalProcessorParametersTests, constructor_shouldSetTheAttributes)
{
    constexpr ProcessingDataType ProcessingDataType = ProcessingDataType::Double;
    constexpr size_t FrameSampleCount = 1;
    constexpr size_t SampleFrequency = 2;
    constexpr size_t InputChannelCount = 3;
    constexpr size_t OutputChannelCount = 4;
    constexpr PcmAudioFrameFormat InputFormat = PcmAudioFrameFormat::Signed8;
    constexpr PcmAudioFrameFormat OutputFormat = PcmAudioFrameFormat::UnsignedPadded24;
    const vector<double>& EqCenterFrequencies{ 1, 2 };
    constexpr size_t MaxOutputDelay = 5;
    constexpr size_t SoundLevelLength = 6;

    SignalProcessorParameters parameters(ProcessingDataType,
        FrameSampleCount,
        SampleFrequency,
        InputChannelCount,
        OutputChannelCount,
        InputFormat,
        OutputFormat,
        EqCenterFrequencies,
        MaxOutputDelay,
        SoundLevelLength);

    EXPECT_EQ(parameters.processingDataType(), ProcessingDataType);
    EXPECT_EQ(parameters.frameSampleCount(), FrameSampleCount);
    EXPECT_EQ(parameters.sampleFrequency(), SampleFrequency);
    EXPECT_EQ(parameters.inputChannelCount(), InputChannelCount);
    EXPECT_EQ(parameters.outputChannelCount(), OutputChannelCount);
    EXPECT_EQ(parameters.inputFormat(), InputFormat);
    EXPECT_EQ(parameters.outputFormat(), OutputFormat);
    EXPECT_EQ(parameters.eqCenterFrequencies(), EqCenterFrequencies);
    EXPECT_EQ(parameters.maxOutputDelay(), MaxOutputDelay);
    EXPECT_EQ(parameters.soundLevelLength(), SoundLevelLength);
}
