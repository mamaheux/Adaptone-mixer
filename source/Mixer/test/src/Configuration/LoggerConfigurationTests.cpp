#include <Mixer/Configuration/LoggerConfiguration.h>

#include <Utils/Exception/InvalidValueException.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(LoggerConfigurationTests, constructor_consoleType_shouldSetTheTypeRelatedAttributes)
{
    LoggerConfiguration configuration(Properties({{"logger.type", "console"}, {"logger.filename", "log.txt"}}));

    EXPECT_EQ(configuration.type(), LoggerConfiguration::Type::Console);
    EXPECT_EQ(configuration.filename(), "");
}

TEST(LoggerConfigurationTests, constructor_fileType_shouldSetTheTypeRelatedAttributes)
{
    LoggerConfiguration configuration(Properties({{"logger.type", "file"}, {"logger.filename", "log.txt"}}));

    EXPECT_EQ(configuration.type(), LoggerConfiguration::Type::File);
    EXPECT_EQ(configuration.filename(), "log.txt");
}

TEST(LoggerConfigurationTests, constructor_invalidType_shouldSetTheTypeRelatedAttributes)
{
    EXPECT_THROW(LoggerConfiguration(Properties({{"logger.type", "bob"}, {"logger.filename", "log.txt"}})),
                 InvalidValueException);
}