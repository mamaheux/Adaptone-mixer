#include <Communication/Messages/Initialization/InitialParametersCreationMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(InitialParametersCreationMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr size_t id = 10;
    const string name = "a name";

    const vector<size_t> inputChannelIds{ 1, 2, 3 };
    constexpr size_t speakersNumber = 2;
    const vector<size_t> auxiliaryChannelIds{ 4, 5 };

    InitialParametersCreationMessage message(id, name, inputChannelIds, speakersNumber, auxiliaryChannelIds);

    EXPECT_EQ(message.seqId(), 1);

    EXPECT_EQ(message.id(), id);
    EXPECT_EQ(message.name(), name);

    EXPECT_EQ(message.inputChannelIds(), vector<size_t>({ 1, 2, 3 }));
    EXPECT_EQ(message.speakersNumber(), speakersNumber);
    EXPECT_EQ(message.auxiliaryChannelIds(), vector<size_t>({ 4, 5 }));
}

TEST(InitialParametersCreationMessageTests, serialization_shouldSerializaToJson)
{
    constexpr size_t id = 10;
    const string name = "a name";

    const vector<size_t> inputChannelIds{ 1, 2, 3 };
    constexpr size_t speakersNumber = 2;
    const vector<size_t> auxiliaryChannelIds{ 4, 5 };

    InitialParametersCreationMessage message(id, name, inputChannelIds, speakersNumber, auxiliaryChannelIds);
    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 1);

    EXPECT_EQ(serializedMessage.at("data").at("id"), id);
    EXPECT_EQ(serializedMessage.at("data").at("name"), name);

    EXPECT_EQ(serializedMessage.at("data").at("inputChannelIds"), inputChannelIds);
    EXPECT_EQ(serializedMessage.at("data").at("speakersNumber"), speakersNumber);
    EXPECT_EQ(serializedMessage.at("data").at("auxiliaryChannelIds"), auxiliaryChannelIds);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(InitialParametersCreationMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 1,"
        "  \"data\": {"
        "    \"id\": 10,"
        "    \"name\": \"super nom\","
        "    \"inputChannelIds\": [1, 2, 3],"
        "    \"speakersNumber\": 4,"
        "    \"auxiliaryChannelIds\": [4, 5]"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<InitialParametersCreationMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 1);

    EXPECT_EQ(deserializedMessage.id(), 10);
    EXPECT_EQ(deserializedMessage.name(), "super nom");

    EXPECT_EQ(deserializedMessage.inputChannelIds(), vector<size_t>({ 1, 2, 3 }));
    EXPECT_EQ(deserializedMessage.speakersNumber(), 4);
    EXPECT_EQ(deserializedMessage.auxiliaryChannelIds(), vector<size_t>({ 4, 5 }));
}
