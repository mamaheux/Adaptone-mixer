#include <Utils/Configuration/Properties.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(PropertiesTests, construtor_map_shouldCopyTheMap)
{
    Properties properties(
        {
            { "key0", "v1" },
            { "key1", "v2" }
        });

    EXPECT_EQ(properties.get<string>("key0"), "v1");
    EXPECT_EQ(properties.get<string>("key1"), "v2");
}

TEST(PropertiesTests, construtor_file_shouldReadTheProperties)
{
    Properties properties("resources/PropertiesTests/valid.properties");

    EXPECT_EQ(properties.get<string>("key0"), "abc");
    EXPECT_EQ(properties.get<string>("key1"), "abc");
    EXPECT_EQ(properties.get<string>("key3"), "abc");

    EXPECT_EQ(properties.get<string>("key4"), "");

    EXPECT_THROW(properties.get<string>("key5"), PropertyNotFoundException);

    EXPECT_THROW(properties.get<int>("key4"), PropertyParseException);

    EXPECT_EQ(properties.get<int>("key_array[0]"), 10);
    EXPECT_EQ(properties.get<int>("key_array[1]"), 10);

    EXPECT_EQ(properties.get<double>("key_array[1]"), 10.5);
}
