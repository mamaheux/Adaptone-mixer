#include <Communication/Messages/Initialization/SymmetryConfirmationMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(SymmetryConfirmationMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr size_t symmetry = 1;
    SymmetryConfirmationMessage message(symmetry);

    EXPECT_EQ(message.seqId(), 5);
    EXPECT_EQ(message.symmetry(), symmetry);
}

TEST(SymmetryConfirmationMessageTests, serialization_shouldSerializaToJson)
{
    constexpr size_t symmetry = 1;
    SymmetryConfirmationMessage message(symmetry);
    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 5);
    EXPECT_EQ(serializedMessage.at("data").at("symmetry"), symmetry);
    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(SymmetryConfirmationMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 5,"
        "  \"data\": {"
        "    \"symmetry\": 0"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<SymmetryConfirmationMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 5);
    EXPECT_EQ(deserializedMessage.symmetry(), 0);
}
