#include <Communication/Messages/Initialization/ConfigurationPosition.h>

#include <gtest/gtest.h>

#include <string>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ConfigurationPositionTests, constructor_shouldSetTheAttributes)
{
    constexpr double X = 10;
    constexpr double Y = 12;
    constexpr PositionType Type = PositionType::Speaker;
    ConfigurationPosition position(X, Y, Type);

    EXPECT_EQ(position.x(), X);
    EXPECT_EQ(position.y(), Y);
    EXPECT_EQ(position.type(), Type);
}

TEST(ConfigurationPositionTests, serialization_shouldSerializaToJson)
{
    constexpr double X = 10;
    constexpr double Y = 12;
    constexpr PositionType Type1 = PositionType::Speaker;
    constexpr PositionType Type2 = PositionType::Probe;
    ConfigurationPosition Position1(10, 12, Type1);
    ConfigurationPosition Position2(X, Y, Type2);

    json serializedPosition1 = Position1;
    json serializedPosition2 = Position2;

    EXPECT_EQ(serializedPosition1.at("x"), X);
    EXPECT_EQ(serializedPosition1.at("y"), Y);
    EXPECT_EQ(serializedPosition1.at("type"), "s");

    EXPECT_EQ(serializedPosition2.at("x"), X);
    EXPECT_EQ(serializedPosition2.at("y"), Y);
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
