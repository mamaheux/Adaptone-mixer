#include <Communication/Messages/Input/ChangeAuxiliaryOutputEqGainsMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ChangeAuxiliaryOutputEqGainsMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr size_t ChannelId = 2;
    const vector<double> Gains{ 1, 10 };
    ChangeAuxiliaryOutputEqGainsMessage message(ChannelId, Gains);

    EXPECT_EQ(message.seqId(), 18);

    EXPECT_EQ(message.channelId(), ChannelId);
    EXPECT_EQ(message.gains(), Gains);
}

TEST(ChangeAuxiliaryOutputEqGainsMessageTests, serialization_shouldSerializaToJson)
{
    constexpr size_t ChannelId = 2;
    const vector<double> Gains{ 1, 10 };
    ChangeAuxiliaryOutputEqGainsMessage message(ChannelId, Gains);

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 18);

    EXPECT_EQ(serializedMessage.at("data").at("channelId"), ChannelId);
    EXPECT_EQ(serializedMessage.at("data").at("gains"), Gains);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(ChangeAuxiliaryOutputEqGainsMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 18,"
        "  \"data\": {"
        "    \"channelId\": 0,"
        "    \"gains\": [1.0, 10.0, 100.0]"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<ChangeAuxiliaryOutputEqGainsMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 18);

    EXPECT_EQ(deserializedMessage.channelId(), 0);
    EXPECT_EQ(deserializedMessage.gains(), vector<double>({ 1, 10, 100 }));
}
