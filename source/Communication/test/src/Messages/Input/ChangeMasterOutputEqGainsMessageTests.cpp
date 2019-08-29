#include <Communication/Messages/Input/ChangeMasterOutputEqGainsMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ChangeMasterOutputEqGainsMessageTests, constructor_shouldSetTheAttributes)
{
    const vector<double> gains{ 1, 10 };
    ChangeMasterOutputEqGainsMessage message(gains);

    EXPECT_EQ(message.seqId(), 15);

    EXPECT_EQ(message.gains(), gains);
}

TEST(ChangeMasterOutputEqGainsMessageTests, serialization_shouldSerializaToJson)
{
    const vector<double> gains{ 1, 10 };
    ChangeMasterOutputEqGainsMessage message(gains);

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 15);

    EXPECT_EQ(serializedMessage.at("data").at("gains"), gains);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(ChangeMasterOutputEqGainsMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 15,"
        "  \"data\": {"
        "    \"gains\": [1.0, 10.0, 100.0]"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<ChangeMasterOutputEqGainsMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 15);

    EXPECT_EQ(deserializedMessage.gains(), vector<double>({ 1, 10, 100 }));
}
