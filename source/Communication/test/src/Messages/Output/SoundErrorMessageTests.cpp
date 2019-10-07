#include <Communication/Messages/Output/SoundErrorMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(SoundErrorMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr double X = 10;
    constexpr double Y = 12;
    constexpr PositionType Type = PositionType::Speaker;
    constexpr double ErrorRate = 1;
    const vector<SoundErrorPosition> positions{ SoundErrorPosition(X, Y, Type, ErrorRate) };

    SoundErrorMessage message(positions);

    EXPECT_EQ(message.seqId(), 21);

    ASSERT_EQ(message.positions().size(), 1);
    EXPECT_EQ(message.positions()[0].x(), X);
    EXPECT_EQ(message.positions()[0].y(), Y);
    EXPECT_EQ(message.positions()[0].type(), Type);
    EXPECT_EQ(message.positions()[0].errorRate(), ErrorRate);
}

TEST(SoundErrorMessageTests, serialization_shouldSerializaToJson)
{
    constexpr double X = 10;
    constexpr double Y = 12;
    constexpr PositionType Type = PositionType::Speaker;
    constexpr double ErrorRate = 1;
    const vector<SoundErrorPosition> positions{ SoundErrorPosition(X, Y, Type, ErrorRate) };

    SoundErrorMessage message(positions);
    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 21);

    ASSERT_EQ(serializedMessage.at("data").at("positions").size(), 1);
    EXPECT_EQ(serializedMessage.at("data").at("positions")[0].at("x"), X);
    EXPECT_EQ(serializedMessage.at("data").at("positions")[0].at("y"), Y);
    EXPECT_EQ(serializedMessage.at("data").at("positions")[0].at("type"), "s");
    EXPECT_EQ(serializedMessage.at("data").at("positions")[0].at("errorRate"), 1);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(SoundErrorMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 21,"
        "  \"data\": {"
        "    \"positions\": ["
        "      {"
        "        \"x\": 140,"
        "        \"y\": 340,"
        "        \"type\": \"s\","
        "        \"errorRate\": 1"
        "      }"
        "    ]"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<SoundErrorMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 21);

    ASSERT_EQ(deserializedMessage.positions().size(), 1);
    EXPECT_EQ(deserializedMessage.positions()[0].x(), 140);
    EXPECT_EQ(deserializedMessage.positions()[0].y(), 340);
    EXPECT_EQ(deserializedMessage.positions()[0].type(), PositionType::Speaker);
    EXPECT_EQ(deserializedMessage.positions()[0].errorRate(), 1);
}
