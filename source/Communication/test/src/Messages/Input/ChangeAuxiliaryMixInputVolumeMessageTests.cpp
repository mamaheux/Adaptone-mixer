#include <Communication/Messages/Input/ChangeAuxiliaryMixInputVolumeMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ChangeAuxiliaryMixInputVolumeMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr size_t channelId = 1;
    constexpr size_t auxiliaryId = 2;
    constexpr double gain = 10;
    ChangeAuxiliaryMixInputVolumeMessage message(channelId, auxiliaryId, gain);

    EXPECT_EQ(message.seqId(), 14);

    EXPECT_EQ(message.channelId(), channelId);
    EXPECT_EQ(message.auxiliaryId(), auxiliaryId);
    EXPECT_EQ(message.gain(), gain);
}

TEST(ChangeAuxiliaryMixInputVolumeMessageTests, serialization_shouldSerializaToJson)
{
    constexpr size_t channelId = 1;
    constexpr size_t auxiliaryId = 2;
    constexpr double gain = 10;
    ChangeAuxiliaryMixInputVolumeMessage message(channelId, auxiliaryId, gain);

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 14);

    EXPECT_EQ(serializedMessage.at("data").at("channelId"), channelId);
    EXPECT_EQ(serializedMessage.at("data").at("auxiliaryId"), auxiliaryId);
    EXPECT_EQ(serializedMessage.at("data").at("gain"), gain);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(ChangeAuxiliaryMixInputVolumeMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 14,"
        "  \"data\": {"
        "    \"channelId\": 0,"
        "    \"auxiliaryId\": 1,"
        "    \"gain\": 100"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<ChangeAuxiliaryMixInputVolumeMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 14);

    EXPECT_EQ(deserializedMessage.channelId(), 0);
    EXPECT_EQ(deserializedMessage.auxiliaryId(), 1);
    EXPECT_EQ(deserializedMessage.gain(), 100);
}
