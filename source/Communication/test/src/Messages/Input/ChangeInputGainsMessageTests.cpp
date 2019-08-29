#include <Communication/Messages/Input/ChangeInputGainsMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ChangeInputGainsMessageTests, constructor_shouldSetTheAttributes)
{
    const vector<double> gains{ 1, 10 };
    ChangeInputGainsMessage message(gains);

    EXPECT_EQ(message.seqId(), 11);

    EXPECT_EQ(message.gains(), gains);
}

TEST(ChangeInputGainsMessageTests, serialization_shouldSerializaToJson)
{
    const vector<double> gains{ 1, 10 };
    ChangeInputGainsMessage message(gains);

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 11);

    EXPECT_EQ(serializedMessage.at("data").at("gains"), gains);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(ChangeInputGainsMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 11,"
        "  \"data\": {"
        "    \"gains\": [1.0, 10.0, 100.0]"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<ChangeInputGainsMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 11);

    EXPECT_EQ(deserializedMessage.gains(), vector<double>({ 1, 10, 100 }));
}
