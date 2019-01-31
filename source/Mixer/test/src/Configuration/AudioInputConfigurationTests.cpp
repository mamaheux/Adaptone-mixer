#include <Mixer/Configuration/AudioInputConfiguration.h>

#include <Utils/Exception/InvalidValueException.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(AudioInputConfigurationTests, constructor_rawFilType_shouldSetTheTypeRelatedAttributes)
{
    AudioInputConfiguration configuration(Properties(
    {
        {"input.type", "raw_file"},
        {"input.format", "signed_8"},
        {"input.filename", "input.raw"}
    }));

    EXPECT_EQ(configuration.type(), AudioInputConfiguration::Type::RawFile);
    EXPECT_EQ(configuration.format(), PcmAudioFrame::Format::Signed8);
    EXPECT_EQ(configuration.filename(), "input.raw");
}

TEST(AudioInputConfigurationTests, constructor_invalidType_shouldSetTheTypeRelatedAttributes)
{
    EXPECT_THROW(AudioInputConfiguration(Properties(
                 {
                     {"input.type", "other"},
                     {"input.format", "signed_8"},
                     {"input.filename", "input.raw"}
                 })),
                 InvalidValueException);
}