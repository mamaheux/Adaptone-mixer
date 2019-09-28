#include <Communication/Messages/Initialization/OptimizedPositionMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(OptimizedPositionMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr double x = 10;
    constexpr double y = 12;
    constexpr PositionType type = PositionType::Speaker;
    const vector<ConfigurationPosition> positions{ ConfigurationPosition(x, y, type) };

    OptimizedPositionMessage message(positions);

    EXPECT_EQ(message.seqId(), 7);

    EXPECT_EQ(message.positions().size(), 1);
    EXPECT_EQ(message.positions()[0].x(), x);
    EXPECT_EQ(message.positions()[0].y(), y);
    EXPECT_EQ(message.positions()[0].type(), type);
}

TEST(OptimizedPositionMessageTests, serialization_shouldSerializaToJson)
{
    constexpr double x = 10;
    constexpr double y = 12;
    constexpr PositionType type = PositionType::Speaker;
    const vector<ConfigurationPosition> positions{ ConfigurationPosition(x, y, type) };

    OptimizedPositionMessage message(positions);
    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 7);

    EXPECT_EQ(serializedMessage.at("data").at("positions").size(), 1);
    EXPECT_EQ(serializedMessage.at("data").at("positions")[0].at("x"), x);
    EXPECT_EQ(serializedMessage.at("data").at("positions")[0].at("y"), y);
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
