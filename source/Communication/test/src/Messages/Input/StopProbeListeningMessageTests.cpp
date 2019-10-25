#include <Communication/Messages/Input/StopProbeListeningMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(StopProbeListeningMessageTests, constructor_shouldSetTheAttributes)
{
    StopProbeListeningMessage message;

    EXPECT_EQ(message.seqId(), 26);
}

TEST(StopProbeListeningMessageTests, serialization_shouldSerializaToJson)
{
    StopProbeListeningMessage message;

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 26);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(StopProbeListeningMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 26"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<StopProbeListeningMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 26);
}
