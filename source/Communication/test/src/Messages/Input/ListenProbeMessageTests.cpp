#include <Communication/Messages/Input/ListenProbeMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ListenProbeMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr uint32_t ProbeId = 1;
    ListenProbeMessage message(ProbeId);

    EXPECT_EQ(message.seqId(), 25);

    EXPECT_EQ(message.probeId(), ProbeId);
}

TEST(ListenProbeMessageTests, serialization_shouldSerializaToJson)
{
    constexpr uint32_t ProbeId = 1;
    ListenProbeMessage message(ProbeId);

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 25);

    EXPECT_EQ(serializedMessage.at("data").at("probeId"), ProbeId);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(ListenProbeMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 25,"
        "  \"data\": {"
        "    \"probeId\": 5"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<ListenProbeMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 25);

    EXPECT_EQ(deserializedMessage.probeId(), 5);
}
