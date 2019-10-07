#include <Communication/Messages/Input/ChangeAllProcessingParametersMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ChangeAllProcessingParametersMessageTests, constructor_shouldSetTheAttributes)
{
    const vector<InputProcessingParameters> Inputs{ InputProcessingParameters(1, 2, false, true, { 1 }),
        InputProcessingParameters(3, 4, true, false, { 2 }) };
    const MasterProcessingParameters Master(5, false, { ChannelGain(5, 6) }, { 3 });
    const vector<AuxiliaryProcessingParameters> Auxiliaries{
        AuxiliaryProcessingParameters(6, 7, true, { ChannelGain(8, 9) }, { 5 }) };

    const vector<size_t> InputChannelIds{ 1, 2, 3 };
    constexpr size_t SpeakersNumber = 2;
    const vector<size_t> AuxiliaryChannelIds{ 4, 5 };

    ChangeAllProcessingParametersMessage message(Inputs, Master, Auxiliaries, InputChannelIds, SpeakersNumber,
        AuxiliaryChannelIds);

    EXPECT_EQ(message.seqId(), 24);

    EXPECT_EQ(message.inputs(), Inputs);
    EXPECT_EQ(message.master(), Master);
    EXPECT_EQ(message.auxiliaries(), Auxiliaries);
    EXPECT_EQ(message.inputChannelIds(), InputChannelIds);
    EXPECT_EQ(message.speakersNumber(), SpeakersNumber);
    EXPECT_EQ(message.auxiliaryChannelIds(), AuxiliaryChannelIds);
}

TEST(ChangeAllProcessingParametersMessageTests, serialization_shouldSerializaToJson)
{
    const vector<InputProcessingParameters> Inputs{ InputProcessingParameters(1, 2, false, true, { 1 }),
        InputProcessingParameters(3, 4, true, false, { 2 }) };
    const MasterProcessingParameters Master(5, false, { ChannelGain(5, 6) }, { 3 });
    const vector<AuxiliaryProcessingParameters> Auxiliaries{
        AuxiliaryProcessingParameters(6, 7, true, { ChannelGain(8, 9) }, { 5 }) };

    const vector<size_t> InputChannelIds{ 1, 2, 3 };
    constexpr size_t SpeakersNumber = 2;
    const vector<size_t> AuxiliaryChannelIds{ 4, 5 };

    ChangeAllProcessingParametersMessage message(Inputs, Master, Auxiliaries, InputChannelIds, SpeakersNumber,
        AuxiliaryChannelIds);

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 24);

    EXPECT_EQ(serializedMessage.at("data").at("channels").at("inputs"), Inputs);
    EXPECT_EQ(static_cast<MasterProcessingParameters>(serializedMessage.at("data").at("channels").at("master")),
        Master);
    EXPECT_EQ(serializedMessage.at("data").at("channels").at("auxiliaries"), Auxiliaries);
    EXPECT_EQ(serializedMessage.at("data").at("channels").at("inputChannelIds"), InputChannelIds);
    EXPECT_EQ(serializedMessage.at("data").at("channels").at("speakersNumber"), SpeakersNumber);
    EXPECT_EQ(serializedMessage.at("data").at("channels").at("auxiliaryChannelIds"), AuxiliaryChannelIds);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(ChangeAllProcessingParametersMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 24,"
        "  \"data\": {"
        "    \"channels\":{"
        "      \"inputs\":["
        "        {"
        "          \"data\": {"
        "            \"channelId\":1,"
        "            \"gain\":2,"
        "            \"isMuted\":false,"
        "            \"isSolo\":true,"
        "            \"eqGains\": [1]"
        "          }"
        "        },"
        "        {"
        "          \"data\": {"
        "            \"channelId\":2,"
        "            \"gain\":3,"
        "            \"isMuted\":true,"
        "            \"isSolo\":false,"
        "            \"eqGains\": [2]"
        "          }"
        "        }"
        "      ],"
        "      \"master\":{"
        "        \"data\": {"
        "          \"gain\":3,"
        "          \"isMuted\":false,"
        "          \"inputs\":["
        "            {"
        "              \"data\": {"
        "                \"channelId\":1,"
        "                \"gain\":2"
        "              }"
        "            },"
        "            {"
        "              \"data\": {"
        "                \"channelId\":2,"
        "                \"gain\":3"
        "              }"
        "            }"
        "          ],"
        "          \"eqGains\": [3]"
        "        }"
        "      },"
        "      \"auxiliaries\":["
        "        {"
        "          \"data\": {"
        "            \"channelId\":10,"
        "            \"gain\":5,"
        "            \"isMuted\":false,"
        "            \"inputs\":["
        "              {"
        "                \"data\": {"
        "                  \"channelId\":1,"
        "                  \"gain\":0"
        "                }"
        "              },"
        "              {"
        "                \"data\": {"
        "                  \"channelId\":2,"
        "                  \"gain\":2"
        "                }"
        "              }"
        "            ],"
        "            \"eqGains\": [4]"
        "          }"
        "        }"
        "      ]"
        "    },"
        "    \"inputChannelIds\": [1, 2, 3],"
        "    \"speakersNumber\": 2,"
        "    \"auxiliaryChannelIds\": [4, 5]"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<ChangeAllProcessingParametersMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 24);

    const vector<InputProcessingParameters> ExpectedInputs{ InputProcessingParameters(1, 2, false, true, { 1 }),
        InputProcessingParameters(2, 3, true, false, { 2 }) };
    const MasterProcessingParameters ExpectedMaster(3, false, { ChannelGain(1, 2), ChannelGain(2, 3) }, { 3 });
    const vector<AuxiliaryProcessingParameters> ExpectedAuxiliaries{
        AuxiliaryProcessingParameters(10, 5, false, { ChannelGain(1, 0), ChannelGain(2, 2) }, { 4 }) };

    const vector<size_t> ExpectedInputChannelIds{ 1, 2, 3 };
    constexpr size_t ExpectedSpeakersNumber = 2;
    const vector<size_t> ExpectedAuxiliaryChannelIds{ 4, 5 };

    ASSERT_EQ(deserializedMessage.inputs(), ExpectedInputs);
    EXPECT_EQ(deserializedMessage.master(), ExpectedMaster);
    EXPECT_EQ(deserializedMessage.auxiliaries(), ExpectedAuxiliaries);
    EXPECT_EQ(deserializedMessage.inputChannelIds(), ExpectedInputChannelIds);
    EXPECT_EQ(deserializedMessage.speakersNumber(), ExpectedSpeakersNumber);
    EXPECT_EQ(deserializedMessage.auxiliaryChannelIds(), ExpectedAuxiliaryChannelIds);
}
