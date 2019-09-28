#include <Communication/Messages/Input/ChangeInputEqGainsMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ChangeInputEqGainsMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr size_t ChannelId = 1;
    const vector<double> Gains{ 1, 10 };
    ChangeInputEqGainsMessage message(ChannelId, Gains);

    EXPECT_EQ(message.seqId(), 12);

    EXPECT_EQ(message.channelId(), ChannelId);
    EXPECT_EQ(message.gains(), Gains);
}

TEST(ChangeInputEqGainsMessageTests, serialization_shouldSerializaToJson)
{
    constexpr size_t ChannelId = 1;
    const vector<double> Gains{ 1, 10 };
    ChangeInputEqGainsMessage message(ChannelId, Gains);

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 12);

    EXPECT_EQ(serializedMessage.at("data").at("channelId"), ChannelId);
    EXPECT_EQ(serializedMessage.at("data").at("gains"), Gains);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(ChangeInputEqGainsMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 12,"
        "  \"data\": {"
        "    \"channelId\": 1,"
        "    \"gains\": [1.0, 10.0, 100.0]"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<ChangeInputEqGainsMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 12);

    EXPECT_EQ(deserializedMessage.channelId(), 1);
    EXPECT_EQ(deserializedMessage.gains(), vector<double>({ 1, 10, 100 }));
}
