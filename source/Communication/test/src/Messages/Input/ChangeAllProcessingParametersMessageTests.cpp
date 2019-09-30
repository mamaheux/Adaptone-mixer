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
    ChangeAllProcessingParametersMessage message(Inputs, Master, Auxiliaries);

    EXPECT_EQ(message.seqId(), 24);

    EXPECT_EQ(message.inputs(), Inputs);
    EXPECT_EQ(message.master(), Master);
    EXPECT_EQ(message.auxiliaries(), Auxiliaries);
}

TEST(ChangeAllProcessingParametersMessageTests, serialization_shouldSerializaToJson)
{
    const vector<InputProcessingParameters> Inputs{ InputProcessingParameters(1, 2, false, true, { 1 }),
        InputProcessingParameters(3, 4, true, false, { 2 }) };
    const MasterProcessingParameters Master(5, false, { ChannelGain(5, 6) }, { 3 });
    const vector<AuxiliaryProcessingParameters> Auxiliaries{
        AuxiliaryProcessingParameters(6, 7, true, { ChannelGain(8, 9) }, { 5 }) };
    ChangeAllProcessingParametersMessage message(Inputs, Master, Auxiliaries);

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 24);

    EXPECT_EQ(serializedMessage.at("data").at("channels").at("inputs"), Inputs);
    EXPECT_EQ(static_cast<MasterProcessingParameters>(serializedMessage.at("data").at("channels").at("master")),
        Master);
    EXPECT_EQ(serializedMessage.at("data").at("channels").at("auxiliaries"), Auxiliaries);

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
        "          \"channelId\":1,"
        "          \"gain\":2,"
        "          \"isMuted\":false,"
        "          \"isSolo\":true,"
        "          \"eqGains\": [1]"
        "        },"
        "        {"
        "          \"channelId\":2,"
        "          \"gain\":3,"
        "          \"isMuted\":true,"
        "          \"isSolo\":false,"
        "          \"eqGains\": [2]"
        "        }"
        "      ],"
        "      \"master\":{"
        "        \"gain\":3,"
        "        \"isMuted\":false,"
        "        \"inputs\":["
        "          {"
        "            \"channelId\":1,"
        "            \"gain\":2"
        "          },"
        "          {"
        "            \"channelId\":2,"
        "            \"gain\":3"
        "          }"
        "        ],"
        "        \"eqGains\": [3]"
        "      },"
        "      \"auxiliaries\":["
        "        {"
        "          \"auxiliaryChannelId\":10,"
        "          \"gain\":5,"
        "          \"isMuted\":false,"
        "          \"inputs\":["
        "            {"
        "              \"channelId\":1,"
        "              \"gain\":0"
        "            },"
        "            {"
        "              \"channelId\":2,"
        "              \"gain\":2"
        "            }"
        "          ],"
        "          \"eqGains\": [4]"
        "        }"
        "      ]"
        "    }"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<ChangeAllProcessingParametersMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 24);

    const vector<InputProcessingParameters> ExpectedInputs{ InputProcessingParameters(1, 2, false, true, { 1 }),
        InputProcessingParameters(2, 3, true, false, { 2 }) };
    const MasterProcessingParameters ExpectedMaster(3, false, { ChannelGain(1, 2), ChannelGain(2, 3) }, { 3 });
    const vector<AuxiliaryProcessingParameters> ExpectedAuxiliaries{
        AuxiliaryProcessingParameters(10, 5, false, { ChannelGain(1, 0), ChannelGain(2, 2) }, { 4 }) };

    ASSERT_EQ(deserializedMessage.inputs(), ExpectedInputs);
    EXPECT_EQ(deserializedMessage.master(), ExpectedMaster);
    EXPECT_EQ(deserializedMessage.auxiliaries(), ExpectedAuxiliaries);
}
