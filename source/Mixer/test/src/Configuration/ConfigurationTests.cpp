#include <Mixer/Configuration/Configuration.h>
#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(ConfigurationTests, constructor_shouldInitializeSubConfigurations)
{
    Configuration configuration(Properties(
    {
        {"logger.type", "console"},
        {"logger.filename", "log.txt"},

        {"audio.frame_sample_count", "32"},
        {"audio.sample_frequency", "48000"},
        {"audio.input_channel_count", "16"},
        {"audio.output_channel_count", "14"},
        {"audio.processing_data_type", "double"},

        {"input.type", "raw_file"},
        {"input.format", "signed_8"},
        {"input.filename", "input.raw"}
    }));

    EXPECT_EQ(configuration.logger().type(), LoggerConfiguration::Type::Console);
    EXPECT_EQ(configuration.logger().filename(), "");

    EXPECT_EQ(configuration.audio().frameSampleCount(), 32);
    EXPECT_EQ(configuration.audio().sampleFrequency(), 48000);
    EXPECT_EQ(configuration.audio().inputChannelCount(), 16);
    EXPECT_EQ(configuration.audio().outputChannelCount(), 14);
    EXPECT_EQ(configuration.audio().processingDataType(), AudioConfiguration::ProcessingDataType::Double);

    EXPECT_EQ(configuration.audioInput().type(), AudioInputConfiguration::Type::RawFile);
    EXPECT_EQ(configuration.audioInput().format(), RawAudioFrame::Format::Signed8);
    EXPECT_EQ(configuration.audioInput().filename(), "input.raw");
}