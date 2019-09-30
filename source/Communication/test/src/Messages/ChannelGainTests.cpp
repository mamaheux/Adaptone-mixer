#include <Communication/Messages/ChannelGain.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ChannelGainTests, constructor_shouldSetTheAttributes)
{
    constexpr size_t ChannelId = 0;
    constexpr double Gain = 1;
    ChannelGain channelGain(ChannelId, Gain);

    EXPECT_EQ(channelGain.channelId(), ChannelId);
    EXPECT_EQ(channelGain.gain(), Gain);
}

TEST(ChannelGainTests, serialization_shouldSerializaToJson)
{
    constexpr size_t ChannelId = 0;
    constexpr double Gain = 1;
    ChannelGain channelGain(ChannelId, Gain);

    json serializedChannelGain = channelGain;

    EXPECT_EQ(serializedChannelGain.at("channelId"), ChannelId);
    EXPECT_EQ(serializedChannelGain.at("gain"), Gain);
}

TEST(ChannelGainTests, deserialization_shouldDeserializeFromJson)
{
    string serializedChannelGain = "{"
        "  \"channelId\": 1,"
        "  \"gain\" : 2"
        "}";

    auto channelGain = json::parse(serializedChannelGain).get<ChannelGain>();

    EXPECT_EQ(channelGain.channelId(), 1);
    EXPECT_EQ(channelGain.gain(), 2);
}
