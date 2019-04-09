#include <Mixer/Configuration/Configuration.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(ConfigurationTests, constructor_shouldInitializeSubConfigurations)
{
    Configuration configuration(Properties(
    {
        { "logger.type", "console" },
        { "logger.filename", "log.txt" },

        { "audio.frame_sample_count", "32" },
        { "audio.sample_frequency", "48000" },
        { "audio.input_channel_count", "16" },
        { "audio.output_channel_count", "14" },
        { "audio.processing_data_type", "double" },
        { "audio.eq.parametric_filter_count", "5" },
        { "audio.eq.center_frequencies", "[10, 20]" },
        { "audio.analysis.sound_level_length", "4096" },

        { "audio.input.type", "raw_file" },
        { "audio.input.format", "signed_8" },
        { "audio.input.filename", "input.raw" },
        { "audio.input.looping", "false" },

        { "audio.output.type", "raw_file" },
        { "audio.output.format", "signed_8" },
        { "audio.output.filename", "output.raw" }
    }));

    EXPECT_EQ(configuration.logger().type(), LoggerConfiguration::Type::Console);
    EXPECT_EQ(configuration.logger().filename(), "");

    EXPECT_EQ(configuration.audio().frameSampleCount(), 32);
    EXPECT_EQ(configuration.audio().sampleFrequency(), 48000);
    EXPECT_EQ(configuration.audio().inputChannelCount(), 16);
    EXPECT_EQ(configuration.audio().outputChannelCount(), 14);
    EXPECT_EQ(configuration.audio().processingDataType(), ProcessingDataType::Double);
    EXPECT_EQ(configuration.audio().parametricEqFilterCount(), 5);
    EXPECT_EQ(configuration.audio().eqCenterFrequencies(), vector<double>({ 10, 20 }));
    EXPECT_EQ(configuration.audio().soundLevelLength(), 4096);

    EXPECT_EQ(configuration.audioInput().type(), AudioInputConfiguration::Type::RawFile);
    EXPECT_EQ(configuration.audioInput().format(), PcmAudioFrame::Format::Signed8);
    EXPECT_EQ(configuration.audioInput().filename(), "input.raw");
    EXPECT_EQ(configuration.audioInput().looping(), false);

    EXPECT_EQ(configuration.audioOutput().type(), AudioOutputConfiguration::Type::RawFile);
    EXPECT_EQ(configuration.audioOutput().format(), PcmAudioFrame::Format::Signed8);
    EXPECT_EQ(configuration.audioOutput().filename(), "output.raw");
}
