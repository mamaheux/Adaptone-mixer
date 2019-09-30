#include <Communication/Messages/Input/ChangeMasterMixInputVolumeMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ChangeMasterMixInputVolumeMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr size_t ChannelId = 1;
    constexpr double Gain = 10;
    ChangeMasterMixInputVolumeMessage message(ChannelId, Gain);

    EXPECT_EQ(message.seqId(), 13);

    EXPECT_EQ(message.channelId(), ChannelId);
    EXPECT_EQ(message.gain(), Gain);
}

TEST(ChangeMasterMixInputVolumeMessageTests, serialization_shouldSerializaToJson)
{
    constexpr size_t ChannelId = 1;
    constexpr double Gain = 10;
    ChangeMasterMixInputVolumeMessage message(ChannelId, Gain);

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 13);

    EXPECT_EQ(serializedMessage.at("data").at("channelId"), ChannelId);
    EXPECT_EQ(serializedMessage.at("data").at("gain"), Gain);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(ChangeMasterMixInputVolumeMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 13,"
        "  \"data\": {"
        "    \"channelId\": 0,"
        "    \"gain\": 100"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<ChangeMasterMixInputVolumeMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 13);

    EXPECT_EQ(deserializedMessage.channelId(), 0);
    EXPECT_EQ(deserializedMessage.gain(), 100);
}
