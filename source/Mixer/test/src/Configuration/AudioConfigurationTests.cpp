#include <Mixer/Configuration/AudioConfiguration.h>

#include <Utils/Exception/InvalidValueException.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(AudioConfigurationTests, constructor_floatProcessingDataType_shouldSetTheAttributes)
{
    AudioConfiguration configuration(Properties(
    {
        { "audio.frame_sample_count", "32" },
        { "audio.sample_frequency", "48000" },
        { "audio.input_channel_count", "16" },
        { "audio.output_channel_count", "14" },
        { "audio.processing_data_type", "float" },
        { "audio.eq.center_frequencies", "[10, 20]" }
    }));

    EXPECT_EQ(configuration.frameSampleCount(), 32);
    EXPECT_EQ(configuration.sampleFrequency(), 48000);
    EXPECT_EQ(configuration.inputChannelCount(), 16);
    EXPECT_EQ(configuration.outputChannelCount(), 14);
    EXPECT_EQ(configuration.processingDataType(), ProcessingDataType::Float);
    EXPECT_EQ(configuration.eqCenterFrequencies(), vector<double>({ 10, 20 }));
}

TEST(AudioConfigurationTests, constructor_doubleProcessingDataType_shouldSetTheAttributes)
{
    AudioConfiguration configuration(Properties(
    {
        { "audio.frame_sample_count", "32" },
        { "audio.sample_frequency", "48000" },
        { "audio.input_channel_count", "16" },
        { "audio.output_channel_count", "14" },
        { "audio.processing_data_type", "double" },
        { "audio.eq.center_frequencies", "[10, 20]" }
    }));

    EXPECT_EQ(configuration.frameSampleCount(), 32);
    EXPECT_EQ(configuration.sampleFrequency(), 48000);
    EXPECT_EQ(configuration.inputChannelCount(), 16);
    EXPECT_EQ(configuration.outputChannelCount(), 14);
    EXPECT_EQ(configuration.processingDataType(), ProcessingDataType::Double);
    EXPECT_EQ(configuration.eqCenterFrequencies(), vector<double>({ 10, 20 }));
}

TEST(AudioConfigurationTests, constructor_invalidProcessingDataType_shouldSetTheAttributes)
{
    EXPECT_THROW(AudioConfiguration configuration(Properties(
        {
            { "audio.frame_sample_count", "32" },
            { "audio.sample_frequency", "48000" },
            { "audio.input_channel_count", "16" },
            { "audio.output_channel_count", "14" },
            { "audio.processing_data_type", "asdasd" }
        })),
        InvalidValueException);
}
