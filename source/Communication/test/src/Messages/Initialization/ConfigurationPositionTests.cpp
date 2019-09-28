#include <Communication/Messages/Initialization/ConfigurationPosition.h>

#include <gtest/gtest.h>

#include <string>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ConfigurationPositionTests, constructor_shouldSetTheAttributes)
{
    constexpr double x = 10;
    constexpr double y = 12;
    constexpr PositionType type = PositionType::Speaker;
    ConfigurationPosition position(x, y, type);

    EXPECT_EQ(position.x(), x);
    EXPECT_EQ(position.y(), y);
    EXPECT_EQ(position.type(), type);
}

TEST(ConfigurationPositionTests, serialization_shouldSerializaToJson)
{
    constexpr double x = 10;
    constexpr double y = 12;
    constexpr PositionType type1 = PositionType::Speaker;
    constexpr PositionType type2 = PositionType::Probe;
    ConfigurationPosition position1(10, 12, type1);
    ConfigurationPosition position2(x, y, type2);

    json serializedPosition1 = position1;
    json serializedPosition2 = position2;

    EXPECT_EQ(serializedPosition1.at("x"), x);
    EXPECT_EQ(serializedPosition1.at("y"), y);
    EXPECT_EQ(serializedPosition1.at("type"), "s");

    EXPECT_EQ(serializedPosition2.at("x"), x);
    EXPECT_EQ(serializedPosition2.at("y"), y);
    EXPECT_EQ(serializedPosition2.at("type"), "m");
}

TEST(ConfigurationPositionTests, deserialization_shouldDeserializeFromJson)
{
    string serializedPosition1 = "{ \"x\": 10, \"y\": 12, \"type\": \"s\" }";
    string serializedPosition2 = "{ \"x\": 10, \"y\": 12, \"type\": \"m\" }";

    auto deserializedPosition1 = json::parse(serializedPosition1).get<ConfigurationPosition>();
    auto deserializedPosition2 = json::parse(serializedPosition2).get<ConfigurationPosition>();

    EXPECT_EQ(deserializedPosition1.x(), 10);
    EXPECT_EQ(deserializedPosition1.y(), 12);
    EXPECT_EQ(deserializedPosition1.type(), PositionType::Speaker);

    EXPECT_EQ(deserializedPosition2.x(), 10);
    EXPECT_EQ(deserializedPosition2.y(), 12);
    EXPECT_EQ(deserializedPosition2.type(), PositionType::Probe);
}
