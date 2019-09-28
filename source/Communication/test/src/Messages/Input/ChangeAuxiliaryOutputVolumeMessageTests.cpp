#include <Communication/Messages/Input/ChangeAuxiliaryOutputVolumeMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ChangeAuxiliaryOutputVolumeMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr size_t ChannelId = 2;
    constexpr double Gain = 10;
    ChangeAuxiliaryOutputVolumeMessage message(ChannelId, Gain);

    EXPECT_EQ(message.seqId(), 20);

    EXPECT_EQ(message.channelId(), ChannelId);
    EXPECT_EQ(message.gain(), Gain);
}

TEST(ChangeAuxiliaryOutputVolumeMessageTests, serialization_shouldSerializaToJson)
{
    constexpr size_t ChannelId = 2;
    constexpr double Gain = 10;
    ChangeAuxiliaryOutputVolumeMessage message(ChannelId, Gain);

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 20);

    EXPECT_EQ(serializedMessage.at("data").at("channelId"), ChannelId);
    EXPECT_EQ(serializedMessage.at("data").at("gain"), Gain);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(ChangeAuxiliaryOutputVolumeMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 20,"
        "  \"data\": {"
        "    \"channelId\": 0,"
        "    \"gain\": 100.0"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<ChangeAuxiliaryOutputVolumeMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 20);

    EXPECT_EQ(deserializedMessage.channelId(), 0);
    EXPECT_EQ(deserializedMessage.gain(), 100);
}
