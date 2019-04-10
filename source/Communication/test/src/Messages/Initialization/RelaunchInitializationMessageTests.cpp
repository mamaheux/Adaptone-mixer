#include <Communication/Messages/Initialization/RelaunchInitializationMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(RelaunchInitializationMessageTests, constructor_shouldSetTheAttributes)
{
    RelaunchInitializationMessage message;
    EXPECT_EQ(message.seqId(), 4);
}

TEST(RelaunchInitializationMessageTests, serialization_shouldSerializaToJson)
{
    RelaunchInitializationMessage message;
    json serializedMessage = message;
    EXPECT_EQ(serializedMessage.at("seqId"), 4);
}

TEST(RelaunchInitializationMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 2"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<RelaunchInitializationMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 4);
}
