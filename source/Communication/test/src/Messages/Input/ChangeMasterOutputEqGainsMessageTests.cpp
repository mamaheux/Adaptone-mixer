#include <Communication/Messages/Input/ChangeMasterOutputEqGainsMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ChangeMasterOutputEqGainsMessageTests, constructor_shouldSetTheAttributes)
{
    const vector<double> Gains{ 1, 10 };
    ChangeMasterOutputEqGainsMessage message(Gains);

    EXPECT_EQ(message.seqId(), 17);

    EXPECT_EQ(message.gains(), Gains);
}

TEST(ChangeMasterOutputEqGainsMessageTests, serialization_shouldSerializaToJson)
{
    const vector<double> Gains{ 1, 10 };
    ChangeMasterOutputEqGainsMessage message(Gains);

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 17);

    EXPECT_EQ(serializedMessage.at("data").at("gains"), Gains);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(ChangeMasterOutputEqGainsMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 17,"
        "  \"data\": {"
        "    \"gains\": [1.0, 10.0, 100.0]"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<ChangeMasterOutputEqGainsMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 17);

    EXPECT_EQ(deserializedMessage.gains(), vector<double>({ 1, 10, 100 }));
}
