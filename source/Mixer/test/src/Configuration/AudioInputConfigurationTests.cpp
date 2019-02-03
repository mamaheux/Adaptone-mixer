#include <Mixer/Configuration/AudioInputConfiguration.h>

#include <Utils/Exception/InvalidValueException.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(AudioInputConfigurationTests, constructor_rawFileType_shouldSetTheTypeRelatedAttributes)
{
    AudioInputConfiguration configuration(Properties(
    {
        {"audio.input.type", "raw_file"},
        {"audio.input.format", "signed_8"},
        {"audio.input.filename", "input.raw"},
        {"audio.input.looping", "true"}
    }));

    EXPECT_EQ(configuration.type(), AudioInputConfiguration::Type::RawFile);
    EXPECT_EQ(configuration.format(), PcmAudioFrame::Format::Signed8);
    EXPECT_EQ(configuration.filename(), "input.raw");
    EXPECT_TRUE(configuration.looping());
}

TEST(AudioInputConfigurationTests, constructor_invalidType_shouldSetTheTypeRelatedAttributes)
{
    EXPECT_THROW(AudioInputConfiguration(Properties(
                 {
                     {"audio.input.type", "other"},
                     {"audio.input.format", "signed_8"},
                     {"audio.input.filename", "input.raw"}
                 })),
                 InvalidValueException);
}