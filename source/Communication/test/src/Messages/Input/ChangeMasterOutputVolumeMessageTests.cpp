#include <Communication/Messages/Input/ChangeMasterOutputVolumeMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ChangeMasterOutputVolumeMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr double Gain = 10;
    ChangeMasterOutputVolumeMessage message(Gain);

    EXPECT_EQ(message.seqId(), 19);

    EXPECT_EQ(message.gain(), Gain);
}

TEST(ChangeMasterOutputVolumeMessageTests, serialization_shouldSerializaToJson)
{
    constexpr double Gain = 10;
    ChangeMasterOutputVolumeMessage message(Gain);

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 19);

    EXPECT_EQ(serializedMessage.at("data").at("gain"), Gain);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(ChangeMasterOutputVolumeMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 19,"
        "  \"data\": {"
        "    \"gain\": 100.0"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<ChangeMasterOutputVolumeMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 19);

    EXPECT_EQ(deserializedMessage.gain(), 100);
}
