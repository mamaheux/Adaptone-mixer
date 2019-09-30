#include <Communication/Messages/Input/ChangeMasterMixInputVolumesMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ChangeMasterMixInputVolumesMessageTests, constructor_shouldSetTheAttributes)
{
    const vector<ChannelGain> Gains{ ChannelGain(1, 1), ChannelGain(2, 10) };
    ChangeMasterMixInputVolumesMessage message(Gains);

    EXPECT_EQ(message.seqId(), 14);

    EXPECT_EQ(message.gains(), Gains);
}

TEST(ChangeMasterMixInputVolumesMessageTests, serialization_shouldSerializaToJson)
{
    const vector<ChannelGain> Gains{ ChannelGain(1, 1), ChannelGain(2, 10) };
    ChangeMasterMixInputVolumesMessage message(Gains);

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 14);

    EXPECT_EQ(serializedMessage.at("data").at("gains"), Gains);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(ChangeMasterMixInputVolumesMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 14,"
        "  \"data\": {"
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

    auto deserializedMessage = json::parse(serializedMessage).get<ChangeMasterMixInputVolumesMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 14);

    EXPECT_EQ(deserializedMessage.gains(), vector<ChannelGain>({ ChannelGain(1, 1), ChannelGain(2, 10) }));
}
