#include <Communication/Messages/Initialization/ReoptimizePositionMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ReoptimizePositionMessageTests, constructor_shouldSetTheAttributes)
{
    ReoptimizePositionMessage message;
    EXPECT_EQ(message.seqId(), 8);
}

TEST(ReoptimizePositionMessageTests, serialization_shouldSerializaToJson)
{
    ReoptimizePositionMessage message;
    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 8);
    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(ReoptimizePositionMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 8"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<ReoptimizePositionMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 8);
}
