#include <Communication/Messages/Output/SoundLevelMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(SoundLevelMessageTests, constructor_shouldSetTheAttributes)
{
    const vector<double> inputAfterGain{ 1, 2 };
    const vector<double> inputAfterEq{ 3, 4 };
    const vector<double> outputAfterGain{ 5, 6 };
    SoundLevelMessage message(inputAfterGain, inputAfterEq, outputAfterGain);

    EXPECT_EQ(message.seqId(), 21);

    EXPECT_EQ(message.inputAfterGain(), inputAfterGain);
    EXPECT_EQ(message.inputAfterEq(), inputAfterEq);
    EXPECT_EQ(message.outputAfterGain(), outputAfterGain);
}

TEST(SoundLevelMessageTests, serialization_shouldSerializaToJson)
{
    const vector<double> inputAfterGain{ 1, 2 };
    const vector<double> inputAfterEq{ 3, 4 };
    const vector<double> outputAfterGain{ 5, 6 };
    SoundLevelMessage message(inputAfterGain, inputAfterEq, outputAfterGain);

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 21);

    EXPECT_EQ(serializedMessage.at("data").at("inputAfterGain"), inputAfterGain);
    EXPECT_EQ(serializedMessage.at("data").at("inputAfterEq"), inputAfterEq);
    EXPECT_EQ(serializedMessage.at("data").at("outputAfterGain"), outputAfterGain);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(SoundLevelMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 21,"
        "  \"data\": {"
        "    \"inputAfterGain\": [1, 2],"
        "    \"inputAfterEq\": [3, 4],"
        "    \"outputAfterGain\": [5, 6]"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<SoundLevelMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 21);

    EXPECT_EQ(deserializedMessage.inputAfterGain(), vector<double>({ 1, 2 }));
    EXPECT_EQ(deserializedMessage.inputAfterEq(), vector<double>({ 3, 4 }));
    EXPECT_EQ(deserializedMessage.outputAfterGain(), vector<double>({ 5, 6 }));
}
