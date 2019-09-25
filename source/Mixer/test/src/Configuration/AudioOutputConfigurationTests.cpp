#include <Mixer/Configuration/AudioOutputConfiguration.h>

#include <Utils/Exception/InvalidValueException.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(AudioOutputConfigurationTests, constructor_rawFileType_shouldSetTheTypeRelatedAttributes)
{
    AudioOutputConfiguration configuration(Properties(
    {
        { "audio.output.type", "raw_file" },
        { "audio.output.format", "signed_8" },
        { "audio.output.filename", "output.raw" }
    }));

    EXPECT_EQ(configuration.type(), AudioOutputConfiguration::Type::RawFile);
    EXPECT_EQ(configuration.format(), PcmAudioFrameFormat::Signed8);
    EXPECT_EQ(configuration.filename(), "output.raw");
}

#if defined(__unix__) || defined(__linux__)

TEST(AudioOutputConfigurationTests, constructor_alsaType_shouldSetTheTypeRelatedAttributes)
{
    AudioOutputConfiguration configuration(Properties(
    {
        { "audio.output.type", "alsa" },
        { "audio.output.format", "signed_8" },
        { "audio.output.device", "hw:0,0" }
    }));

    EXPECT_EQ(configuration.type(), AudioOutputConfiguration::Type::Alsa);
    EXPECT_EQ(configuration.format(), PcmAudioFrameFormat::Signed8);
    EXPECT_EQ(configuration.device(), "hw:0,0");
}

#endif

TEST(AudioOutputConfigurationTests, constructor_invalidType_shouldSetTheTypeRelatedAttributes)
{
    EXPECT_THROW(AudioOutputConfiguration(Properties(
        {
            { "audio.output.type", "other" },
            { "audio.output.format", "signed_8" },
            { "audio.output.filename", "output.raw" }
        })),
        InvalidValueException);
}
