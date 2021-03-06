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
        { "audio.eq.center_frequencies", "[10, 20]" },
        { "audio.max_output_delay", "8192" },
        { "audio.analysis.sound_level_length", "4096" },
        { "audio.analysis.spectrum.fft_length", "2048" },
        { "audio.analysis.spectrum.point_count_per_decade", "10" },
        { "audio.headphone_channel_indexes", "[12, 13]"}
    }));

    EXPECT_EQ(configuration.frameSampleCount(), 32);
    EXPECT_EQ(configuration.sampleFrequency(), 48000);
    EXPECT_EQ(configuration.inputChannelCount(), 16);
    EXPECT_EQ(configuration.outputChannelCount(), 14);
    EXPECT_EQ(configuration.processingDataType(), ProcessingDataType::Float);
    EXPECT_EQ(configuration.eqCenterFrequencies(), vector<double>({ 10, 20 }));
    EXPECT_EQ(configuration.maxOutputDelay(), 8192);
    EXPECT_EQ(configuration.soundLevelLength(), 4096);
    EXPECT_EQ(configuration.spectrumAnalysisFftLength(), 2048);
    EXPECT_EQ(configuration.spectrumAnalysisPointCountPerDecade(), 10);
    EXPECT_EQ(configuration.headphoneChannelIndexes(), vector<size_t>({ 12, 13 }));
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
        { "audio.eq.center_frequencies", "[10, 20]" },
        { "audio.max_output_delay", "8192" },
        { "audio.analysis.sound_level_length", "4096" },
        { "audio.analysis.spectrum.fft_length", "2048" },
        { "audio.analysis.spectrum.point_count_per_decade", "10" },
        { "audio.headphone_channel_indexes", "[12]"}
    }));

    EXPECT_EQ(configuration.frameSampleCount(), 32);
    EXPECT_EQ(configuration.sampleFrequency(), 48000);
    EXPECT_EQ(configuration.inputChannelCount(), 16);
    EXPECT_EQ(configuration.outputChannelCount(), 14);
    EXPECT_EQ(configuration.processingDataType(), ProcessingDataType::Double);
    EXPECT_EQ(configuration.eqCenterFrequencies(), vector<double>({ 10, 20 }));
    EXPECT_EQ(configuration.maxOutputDelay(), 8192);
    EXPECT_EQ(configuration.soundLevelLength(), 4096);
    EXPECT_EQ(configuration.spectrumAnalysisFftLength(), 2048);
    EXPECT_EQ(configuration.spectrumAnalysisPointCountPerDecade(), 10);
    EXPECT_EQ(configuration.headphoneChannelIndexes(), vector<size_t>({ 12 }));
}

TEST(AudioConfigurationTests, constructor_invalidProcessingDataType_shouldSetTheAttributes)
{
    EXPECT_THROW(AudioConfiguration configuration(Properties(
        {
            { "audio.frame_sample_count", "32" },
            { "audio.sample_frequency", "48000" },
            { "audio.input_channel_count", "16" },
            { "audio.output_channel_count", "14" },
            { "audio.processing_data_type", "sdfsdfdsfsd" },
            { "audio.eq.center_frequencies", "[10, 20]" },
            { "audio.max_output_delay", "8192" },
            { "audio.analysis.sound_level_length", "4096" },
            { "audio.analysis.spectrum.fft_length", "2048" },
            { "audio.analysis.spectrum.point_count_per_decade", "10" },
            { "audio.headphone_channel_indexes", "[12, 13]"}
        })),
        InvalidValueException);
}

TEST(AudioConfigurationTests, constructor_invalidSpectrumAnalysisFftLength_shouldSetTheAttributes)
{
    EXPECT_THROW(AudioConfiguration configuration(Properties(
        {
            { "audio.frame_sample_count", "32" },
            { "audio.sample_frequency", "48000" },
            { "audio.input_channel_count", "16" },
            { "audio.output_channel_count", "14" },
            { "audio.processing_data_type", "float" },
            { "audio.eq.center_frequencies", "[10, 20]" },
            { "audio.max_output_delay", "8192" },
            { "audio.analysis.sound_level_length", "4096" },
            { "audio.analysis.spectrum.fft_length", "2047" },
            { "audio.analysis.spectrum.point_count_per_decade", "10" },
            { "audio.headphone_channel_indexes", "[12, 13]"}
        })),
        InvalidValueException);
}

TEST(AudioConfigurationTests, constructor_invalidMaxOutputDelay_shouldSetTheAttributes)
{
    EXPECT_THROW(AudioConfiguration configuration(Properties(
        {
            { "audio.frame_sample_count", "32" },
            { "audio.sample_frequency", "48000" },
            { "audio.input_channel_count", "16" },
            { "audio.output_channel_count", "14" },
            { "audio.processing_data_type", "float" },
            { "audio.eq.center_frequencies", "[10, 20]" },
            { "audio.max_output_delay", "8191" },
            { "audio.analysis.sound_level_length", "4096" },
            { "audio.analysis.spectrum.fft_length", "2047" },
            { "audio.analysis.spectrum.point_count_per_decade", "10" },
            { "audio.headphone_channel_indexes", "[12, 13]"}
        })),
        InvalidValueException);
}

TEST(AudioConfigurationTests, constructor_tooFewHeadphoneChannelIndexCount_shouldSetTheAttributes)
{
    EXPECT_THROW(AudioConfiguration configuration(Properties(
        {
            { "audio.frame_sample_count", "32" },
            { "audio.sample_frequency", "48000" },
            { "audio.input_channel_count", "16" },
            { "audio.output_channel_count", "14" },
            { "audio.processing_data_type", "float" },
            { "audio.eq.center_frequencies", "[10, 20]" },
            { "audio.max_output_delay", "8191" },
            { "audio.analysis.sound_level_length", "4096" },
            { "audio.analysis.spectrum.fft_length", "2047" },
            { "audio.analysis.spectrum.point_count_per_decade", "10" },
            { "audio.headphone_channel_indexes", "[]"}
        })),
        InvalidValueException);
}

TEST(AudioConfigurationTests, constructor_tooManyHeadphoneChannelIndexCount_shouldSetTheAttributes)
{
    EXPECT_THROW(AudioConfiguration configuration(Properties(
        {
            { "audio.frame_sample_count", "32" },
            { "audio.sample_frequency", "48000" },
            { "audio.input_channel_count", "16" },
            { "audio.output_channel_count", "14" },
            { "audio.processing_data_type", "float" },
            { "audio.eq.center_frequencies", "[10, 20]" },
            { "audio.max_output_delay", "8191" },
            { "audio.analysis.sound_level_length", "4096" },
            { "audio.analysis.spectrum.fft_length", "2047" },
            { "audio.analysis.spectrum.point_count_per_decade", "10" },
            { "audio.headphone_channel_indexes", "[10, 11, 12]"}
        })),
        InvalidValueException);
}

TEST(AudioConfigurationTests, constructor_invalidHeadphoneChannelIndex_shouldSetTheAttributes)
{
    EXPECT_THROW(AudioConfiguration configuration(Properties(
        {
            { "audio.frame_sample_count", "32" },
            { "audio.sample_frequency", "48000" },
            { "audio.input_channel_count", "16" },
            { "audio.output_channel_count", "14" },
            { "audio.processing_data_type", "float" },
            { "audio.eq.center_frequencies", "[10, 20]" },
            { "audio.max_output_delay", "8191" },
            { "audio.analysis.sound_level_length", "4096" },
            { "audio.analysis.spectrum.fft_length", "2047" },
            { "audio.analysis.spectrum.point_count_per_decade", "10" },
            { "audio.headphone_channel_indexes", "[16]"}
        })),
        InvalidValueException);
}
