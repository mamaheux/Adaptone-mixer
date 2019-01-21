#include <Mixer/Configuration/Configuration.h>
#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(ConfigurationTests, constructor_shouldInitializeSubConfigurations)
{
    Configuration configuration(Properties({{"logger.type", "console"}, {"logger.filename", "log.txt"}}));

    EXPECT_EQ(configuration.logger().type(), LoggerConfiguration::Type::Console);
    EXPECT_EQ(configuration.logger().filename(), "");
}