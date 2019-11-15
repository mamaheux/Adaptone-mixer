#include <Communication/Messages/Input/ToogleUniformizationMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ToogleUniformizationMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr bool IsOn = true;
    ToogleUniformizationMessage message(IsOn);

    EXPECT_EQ(message.seqId(), 27);

    EXPECT_EQ(message.isOn(), IsOn);
}

TEST(ToogleUniformizationMessageTests, serialization_shouldSerializaToJson)
{
    constexpr bool IsOn = true;
    ToogleUniformizationMessage message(IsOn);

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 27);

    EXPECT_EQ(serializedMessage.at("data").at("isUniformizationOn"), IsOn);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(ToogleUniformizationMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 27,"
        "  \"data\": {"
        "    \"isUniformizationOn\": true"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<ToogleUniformizationMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 27);

    EXPECT_EQ(deserializedMessage.isOn(), true);
}
