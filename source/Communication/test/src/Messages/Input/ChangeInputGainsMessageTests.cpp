#include <Communication/Messages/Input/ChangeInputGainsMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ChangeInputGainsMessageTests, constructor_shouldSetTheAttributes)
{
    const vector<ChannelGain> Gains{ ChannelGain(1, 1), ChannelGain(2, 10) };
    ChangeInputGainsMessage message(Gains);

    EXPECT_EQ(message.seqId(), 11);

    EXPECT_EQ(message.gains(), Gains);
}

TEST(ChangeInputGainsMessageTests, serialization_shouldSerializaToJson)
{
    const vector<ChannelGain> Gains{ ChannelGain(1, 1), ChannelGain(2, 10) };
    ChangeInputGainsMessage message(Gains);

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 11);

    EXPECT_EQ(serializedMessage.at("data").at("gains"), Gains);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(ChangeInputGainsMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 11,"
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

    auto deserializedMessage = json::parse(serializedMessage).get<ChangeInputGainsMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 11);

    EXPECT_EQ(deserializedMessage.gains(), vector<ChannelGain>({ ChannelGain(1, 1), ChannelGain(2, 10) }));
}
