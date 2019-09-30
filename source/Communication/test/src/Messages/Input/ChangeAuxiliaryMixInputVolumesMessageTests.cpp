#include <Communication/Messages/Input/ChangeAuxiliaryMixInputVolumesMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ChangeAuxiliaryMixInputVolumesMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr size_t AuxiliaryChannelId = 2;
    const vector<ChannelGain> Gains{ ChannelGain(1, 1), ChannelGain(2, 10) };
    ChangeAuxiliaryMixInputVolumesMessage message(AuxiliaryChannelId, Gains);

    EXPECT_EQ(message.seqId(), 16);

    EXPECT_EQ(message.auxiliaryChannelId(), AuxiliaryChannelId);
    EXPECT_EQ(message.gains(), Gains);
}

TEST(ChangeAuxiliaryMixInputVolumesMessageTests, serialization_shouldSerializaToJson)
{
    constexpr size_t AuxiliaryChannelId = 2;
    const vector<ChannelGain> Gains{ ChannelGain(1, 1), ChannelGain(2, 10) };
    ChangeAuxiliaryMixInputVolumesMessage message(AuxiliaryChannelId, Gains);

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 16);

    EXPECT_EQ(serializedMessage.at("data").at("auxiliaryChannelId"), AuxiliaryChannelId);
    EXPECT_EQ(serializedMessage.at("data").at("gains"), Gains);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(ChangeAuxiliaryMixInputVolumesMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 16,"
        "  \"data\": {"
        "    \"auxiliaryChannelId\": 1,"
        "    \"gains\": ["
        "      {"
        "         \"channelId\": 1,"
        "         \"gain\" : 1"
        "      },"
        "      {"
        "        \"channelId\": 2,"
        "        \"gain\" : 10"
        "      }"
        "    ]"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<ChangeAuxiliaryMixInputVolumesMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 16);

    EXPECT_EQ(deserializedMessage.auxiliaryChannelId(), 1);
    EXPECT_EQ(deserializedMessage.gains(), vector<ChannelGain>({ ChannelGain(1, 1), ChannelGain(2, 10) }));
}
