#include <Communication/Messages/Input/ChangeAuxiliaryOutputVolumeMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ChangeAuxiliaryOutputVolumeMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr size_t channelId = 2;
    constexpr double gain = 10;
    ChangeAuxiliaryOutputVolumeMessage message(channelId, gain);

    EXPECT_EQ(message.seqId(), 18);

    EXPECT_EQ(message.channelId(), channelId);
    EXPECT_EQ(message.gain(), gain);
}

TEST(ChangeAuxiliaryOutputVolumeMessageTests, serialization_shouldSerializaToJson)
{
    constexpr size_t channelId = 2;
    constexpr double gain = 10;
    ChangeAuxiliaryOutputVolumeMessage message(channelId, gain);

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 18);

    EXPECT_EQ(serializedMessage.at("data").at("channelId"), channelId);
    EXPECT_EQ(serializedMessage.at("data").at("gain"), gain);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(ChangeAuxiliaryOutputVolumeMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 18,"
        "  \"data\": {"
        "    \"channelId\": 0,"
        "    \"gain\": 100.0"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<ChangeAuxiliaryOutputVolumeMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 18);

    EXPECT_EQ(deserializedMessage.channelId(), 0);
    EXPECT_EQ(deserializedMessage.gain(), 100);
}
