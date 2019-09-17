#include <Mixer/Configuration/UniformizationConfiguration.h>

#include <gtest/gtest.h>

using namespace adaptone;

TEST(UniformizationConfigurationTests, constructor_shouldSetTheAttributes)
{
    UniformizationConfiguration configuration(Properties(
    {
        { "dummy1", "dummy" },
        { "dummy2", "dummy" }
    }));

    //TODO Add expect calls
}
