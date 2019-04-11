#include <Communication/Messages/Initialization/LaunchInitializationMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(LaunchInitializationMessageTests, constructor_shouldSetTheAttributes)
{
    LaunchInitializationMessage message;
    EXPECT_EQ(message.seqId(), 2);
}

TEST(LaunchInitializationMessageTests, serialization_shouldSerializaToJson)
{
    LaunchInitializationMessage message;
    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 2);
    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(LaunchInitializationMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 2"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<LaunchInitializationMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 2);
}
