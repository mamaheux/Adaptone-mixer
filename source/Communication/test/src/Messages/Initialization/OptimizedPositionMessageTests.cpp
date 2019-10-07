#include <Communication/Messages/Initialization/OptimizedPositionMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(OptimizedPositionMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr double X = 10;
    constexpr double Y = 12;
    constexpr PositionType Type = PositionType::Speaker;
    const vector<ConfigurationPosition> Positions{ ConfigurationPosition(X, Y, Type) };

    OptimizedPositionMessage message(Positions);

    EXPECT_EQ(message.seqId(), 7);

    EXPECT_EQ(message.positions().size(), 1);
    EXPECT_EQ(message.positions()[0].x(), X);
    EXPECT_EQ(message.positions()[0].y(), Y);
    EXPECT_EQ(message.positions()[0].type(), Type);
}

TEST(OptimizedPositionMessageTests, serialization_shouldSerializaToJson)
{
    constexpr double X = 10;
    constexpr double Y = 12;
    constexpr PositionType Type = PositionType::Speaker;
    const vector<ConfigurationPosition> Positions{ ConfigurationPosition(X, Y, Type) };

    OptimizedPositionMessage message(Positions);
    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 7);

    EXPECT_EQ(serializedMessage.at("data").at("positions").size(), 1);
    EXPECT_EQ(serializedMessage.at("data").at("positions")[0].at("x"), X);
    EXPECT_EQ(serializedMessage.at("data").at("positions")[0].at("y"), Y);
    EXPECT_EQ(serializedMessage.at("data").at("positions")[0].at("type"), "s");

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(OptimizedPositionMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 7,"
        "  \"data\": {"
        "    \"positions\": ["
        "      {"
        "        \"x\": 140,"
        "        \"y\": 340,"
        "        \"type\": \"s\""
        "      }"
        "    ]"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<OptimizedPositionMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 7);

    EXPECT_EQ(deserializedMessage.positions().size(), 1);
    EXPECT_EQ(deserializedMessage.positions()[0].x(), 140);
    EXPECT_EQ(deserializedMessage.positions()[0].y(), 340);
    EXPECT_EQ(deserializedMessage.positions()[0].type(), PositionType::Speaker);
}
