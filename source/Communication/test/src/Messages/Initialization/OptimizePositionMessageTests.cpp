#include <Communication/Messages/Initialization/OptimizePositionMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(OptimizePositionMessageTests, constructor_shouldSetTheAttributes)
{
    OptimizePositionMessage message;
    EXPECT_EQ(message.seqId(), 6);
}

TEST(OptimizePositionMessageTests, serialization_shouldSerializaToJson)
{
    OptimizePositionMessage message;
    json serializedMessage = message;
    EXPECT_EQ(serializedMessage.at("seqId"), 6);
}

TEST(OptimizePositionMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 6"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<OptimizePositionMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 6);
}
