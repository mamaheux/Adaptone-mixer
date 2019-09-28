#include <Communication/Messages/Input/ChangeInputGainMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ChangeInputGainMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr size_t ChannelId = 1;
    constexpr double Gain = 10;
    ChangeInputGainMessage message(ChannelId, Gain);

    EXPECT_EQ(message.seqId(), 10);

    EXPECT_EQ(message.channelId(), ChannelId);
    EXPECT_EQ(message.gain(), Gain);
}

TEST(ChangeInputGainMessageTests, serialization_shouldSerializaToJson)
{
    constexpr size_t ChannelId = 1;
    constexpr double Gain = 10;
    ChangeInputGainMessage message(ChannelId, Gain);

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 10);

    EXPECT_EQ(serializedMessage.at("data").at("channelId"), ChannelId);
    EXPECT_EQ(serializedMessage.at("data").at("gain"), Gain);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(ChangeInputGainMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 10,"
        "  \"data\": {"
        "    \"channelId\": 0,"
        "    \"gain\": 100"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<ChangeInputGainMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 10);

    EXPECT_EQ(deserializedMessage.channelId(), 0);
    EXPECT_EQ(deserializedMessage.gain(), 100);
}
