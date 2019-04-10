#include <Communication/Messages/Initialization/ConfigurationConfirmationMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ConfigurationConfirmationMessageTests, constructor_shouldSetTheAttributes)
{
    ConfigurationConfirmationMessage message;
    EXPECT_EQ(message.seqId(), 9);
}

TEST(ConfigurationConfirmationMessageTests, serialization_shouldSerializaToJson)
{
    ConfigurationConfirmationMessage message;
    json serializedMessage = message;
    EXPECT_EQ(serializedMessage.at("seqId"), 9);
}

TEST(ConfigurationConfirmationMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 9"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<ConfigurationConfirmationMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 9);
}
