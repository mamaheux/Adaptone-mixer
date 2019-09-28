#include <Communication/Messages/Initialization/PositionConfirmationMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(PositionConfirmationMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr double x1 = 10;
    constexpr double y1 = 12;
    constexpr PositionType type1 = PositionType::Speaker;
    const vector<ConfigurationPosition> firstSymmetryPositions{ ConfigurationPosition(x1, y1, type1) };

    constexpr double x2 = 9;
    constexpr double y2 = 5;
    constexpr PositionType type2 = PositionType::Probe;
    const vector<ConfigurationPosition> secondSymmetryPositions{ ConfigurationPosition(x2, y2, type2) };

    PositionConfirmationMessage message(firstSymmetryPositions, secondSymmetryPositions);

    EXPECT_EQ(message.seqId(), 3);

    EXPECT_EQ(message.firstSymmetryPositions().size(), 1);
    EXPECT_EQ(message.firstSymmetryPositions()[0].x(), x1);
    EXPECT_EQ(message.firstSymmetryPositions()[0].y(), y1);
    EXPECT_EQ(message.firstSymmetryPositions()[0].type(), type1);

    EXPECT_EQ(message.secondSymmetryPositions().size(), 1);
    EXPECT_EQ(message.secondSymmetryPositions()[0].x(), x2);
    EXPECT_EQ(message.secondSymmetryPositions()[0].y(), y2);
    EXPECT_EQ(message.secondSymmetryPositions()[0].type(), type2);
}

TEST(PositionConfirmationMessageTests, serialization_shouldSerializaToJson)
{
    constexpr double x1 = 10;
    constexpr double y1 = 12;
    constexpr PositionType type1 = PositionType::Speaker;
    const vector<ConfigurationPosition> firstSymmetryPositions{ ConfigurationPosition(x1, y1, type1) };

    constexpr double x2 = 9;
    constexpr double y2 = 5;
    constexpr PositionType type2 = PositionType::Probe;
    const vector<ConfigurationPosition> secondSymmetryPositions{ ConfigurationPosition(x2, y2, type2) };

    PositionConfirmationMessage message(firstSymmetryPositions, secondSymmetryPositions);
    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 3);

    EXPECT_EQ(serializedMessage.at("data").at("firstSymmetryPositions").size(), 1);
    EXPECT_EQ(serializedMessage.at("data").at("firstSymmetryPositions")[0].at("x"), x1);
    EXPECT_EQ(serializedMessage.at("data").at("firstSymmetryPositions")[0].at("y"), y1);
    EXPECT_EQ(serializedMessage.at("data").at("firstSymmetryPositions")[0].at("type"), "s");

    EXPECT_EQ(serializedMessage.at("data").at("secondSymmetryPositions").size(), 1);
    EXPECT_EQ(serializedMessage.at("data").at("secondSymmetryPositions")[0].at("x"), x2);
    EXPECT_EQ(serializedMessage.at("data").at("secondSymmetryPositions")[0].at("y"), y2);
    EXPECT_EQ(serializedMessage.at("data").at("secondSymmetryPositions")[0].at("type"), "m");

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(PositionConfirmationMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 3,"
        "  \"data\": {"
        "    \"firstSymmetryPositions\": ["
        "      {"
        "        \"x\": 140,"
        "        \"y\": 340,"
        "        \"type\": \"s\""
        "      }"
        "    ],"
        "    \"secondSymmetryPositions\": ["
        "      {"
        "        \"x\": 340,"
        "        \"y\": 140,"
        "        \"type\": \"m\""
        "      }"
        "    ]"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<PositionConfirmationMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 3);

    EXPECT_EQ(deserializedMessage.firstSymmetryPositions().size(), 1);
    EXPECT_EQ(deserializedMessage.firstSymmetryPositions()[0].x(), 140);
    EXPECT_EQ(deserializedMessage.firstSymmetryPositions()[0].y(), 340);
    EXPECT_EQ(deserializedMessage.firstSymmetryPositions()[0].type(), PositionType::Speaker);

    EXPECT_EQ(deserializedMessage.secondSymmetryPositions().size(), 1);
    EXPECT_EQ(deserializedMessage.secondSymmetryPositions()[0].x(), 340);
    EXPECT_EQ(deserializedMessage.secondSymmetryPositions()[0].y(), 140);
    EXPECT_EQ(deserializedMessage.secondSymmetryPositions()[0].type(), PositionType::Probe);
}
