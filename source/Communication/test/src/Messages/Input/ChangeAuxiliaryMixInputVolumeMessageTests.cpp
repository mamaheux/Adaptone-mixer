#include <Communication/Messages/Input/ChangeAuxiliaryMixInputVolumeMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ChangeAuxiliaryMixInputVolumeMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr size_t ChannelId = 1;
    constexpr size_t AuxiliaryChannelId = 2;
    constexpr double Gain = 10;
    ChangeAuxiliaryMixInputVolumeMessage message(ChannelId, AuxiliaryChannelId, Gain);

    EXPECT_EQ(message.seqId(), 15);

    EXPECT_EQ(message.channelId(), ChannelId);
    EXPECT_EQ(message.auxiliaryChannelId(), AuxiliaryChannelId);
    EXPECT_EQ(message.gain(), Gain);
}

TEST(ChangeAuxiliaryMixInputVolumeMessageTests, serialization_shouldSerializaToJson)
{
    constexpr size_t ChannelId = 1;
    constexpr size_t AuxiliaryChannelId = 2;
    constexpr double Gain = 10;
    ChangeAuxiliaryMixInputVolumeMessage message(ChannelId, AuxiliaryChannelId, Gain);

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 15);

    EXPECT_EQ(serializedMessage.at("data").at("channelId"), ChannelId);
    EXPECT_EQ(serializedMessage.at("data").at("auxiliaryChannelId"), AuxiliaryChannelId);
    EXPECT_EQ(serializedMessage.at("data").at("gain"), Gain);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(ChangeAuxiliaryMixInputVolumeMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 15,"
        "  \"data\": {"
        "    \"channelId\": 0,"
        "    \"auxiliaryChannelId\": 1,"
        "    \"gain\": 100"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<ChangeAuxiliaryMixInputVolumeMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 15);

    EXPECT_EQ(deserializedMessage.channelId(), 0);
    EXPECT_EQ(deserializedMessage.auxiliaryChannelId(), 1);
    EXPECT_EQ(deserializedMessage.gain(), 100);
}
