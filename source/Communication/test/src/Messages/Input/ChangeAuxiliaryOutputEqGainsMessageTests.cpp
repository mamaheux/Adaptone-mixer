#include <Communication/Messages/Input/ChangeAuxiliaryOutputEqGainsMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ChangeAuxiliaryOutputEqGainsMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr size_t channelId = 2;
    const vector<double> gains{ 1, 10 };
    ChangeAuxiliaryOutputEqGainsMessage message(channelId, gains);

    EXPECT_EQ(message.seqId(), 16);

    EXPECT_EQ(message.channelId(), channelId);
    EXPECT_EQ(message.gains(), gains);
}

TEST(ChangeAuxiliaryOutputEqGainsMessageTests, serialization_shouldSerializaToJson)
{
    constexpr size_t channelId = 2;
    const vector<double> gains{ 1, 10 };
    ChangeAuxiliaryOutputEqGainsMessage message(channelId, gains);

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 16);

    EXPECT_EQ(serializedMessage.at("data").at("channelId"), channelId);
    EXPECT_EQ(serializedMessage.at("data").at("gains"), gains);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(ChangeAuxiliaryOutputEqGainsMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 16,"
        "  \"data\": {"
        "    \"channelId\": 0,"
        "    \"gains\": [1.0, 10.0, 100.0]"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<ChangeAuxiliaryOutputEqGainsMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 16);

    EXPECT_EQ(deserializedMessage.channelId(), 0);
    EXPECT_EQ(deserializedMessage.gains(), vector<double>({ 1, 10, 100 }));
}
