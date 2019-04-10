#include <Communication/Messages/Initialization/InitialParametersCreationMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(InitialParametersCreationMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr size_t id = 10;
    const string name = "a name";

    constexpr size_t monitorsNumber = 1;
    constexpr size_t speakersNumber = 2;
    constexpr size_t probesNumber = 3;

    InitialParametersCreationMessage message(id, name, monitorsNumber, speakersNumber, probesNumber);

    EXPECT_EQ(message.seqId(), 1);

    EXPECT_EQ(message.id(), id);
    EXPECT_EQ(message.name(), name);

    EXPECT_EQ(message.monitorsNumber(), monitorsNumber);
    EXPECT_EQ(message.speakersNumber(), speakersNumber);
    EXPECT_EQ(message.probesNumber(), probesNumber);
}

TEST(InitialParametersCreationMessageTests, serialization_shouldSerializaToJson)
{
    constexpr size_t id = 10;
    const string name = "a name";

    constexpr size_t monitorsNumber = 1;
    constexpr size_t speakersNumber = 2;
    constexpr size_t probesNumber = 3;

    InitialParametersCreationMessage message(id, name, monitorsNumber, speakersNumber, probesNumber);

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 1);

    EXPECT_EQ(serializedMessage.at("data").at("id"), id);
    EXPECT_EQ(serializedMessage.at("data").at("name"), name);

    EXPECT_EQ(serializedMessage.at("data").at("monitorsNumber"), monitorsNumber);
    EXPECT_EQ(serializedMessage.at("data").at("speakersNumber"), speakersNumber);
    EXPECT_EQ(serializedMessage.at("data").at("probesNumber"), probesNumber);
}

TEST(InitialParametersCreationMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 1,"
        "  \"data\": {"
        "    \"id\": 10,"
        "    \"name\": \"super nom\","
        "    \"monitorsNumber\": 5,"
        "    \"speakersNumber\": 4,"
        "    \"probesNumber\": 8"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<InitialParametersCreationMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 1);

    EXPECT_EQ(deserializedMessage.id(), 10);
    EXPECT_EQ(deserializedMessage.name(), "super nom");

    EXPECT_EQ(deserializedMessage.monitorsNumber(), 5);
    EXPECT_EQ(deserializedMessage.speakersNumber(), 4);
    EXPECT_EQ(deserializedMessage.probesNumber(), 8);
}
