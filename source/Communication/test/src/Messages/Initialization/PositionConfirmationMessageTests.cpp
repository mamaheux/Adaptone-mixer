#include <Communication/Messages/Initialization/PositionConfirmationMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(PositionConfirmationMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr double X1 = 10;
    constexpr double Y1 = 12;
    constexpr PositionType Type1 = PositionType::Speaker;
    constexpr uint32_t Id1 = 2;
    const vector<ConfigurationPosition> FirstSymmetryPositions{ ConfigurationPosition(X1, Y1, Type1, Id1) };

    constexpr double X2 = 9;
    constexpr double Y2 = 5;
    constexpr PositionType Type2 = PositionType::Probe;
    constexpr uint32_t Id2 = 3;
    const vector<ConfigurationPosition> SecondSymmetryPositions{ ConfigurationPosition(X2, Y2, Type2, Id2) };

    PositionConfirmationMessage message(FirstSymmetryPositions, SecondSymmetryPositions);

    EXPECT_EQ(message.seqId(), 3);

    EXPECT_EQ(message.firstSymmetryPositions().size(), 1);
    EXPECT_EQ(message.firstSymmetryPositions()[0].x(), X1);
    EXPECT_EQ(message.firstSymmetryPositions()[0].y(), Y1);
    EXPECT_EQ(message.firstSymmetryPositions()[0].type(), Type1);
    EXPECT_EQ(message.firstSymmetryPositions()[0].id(), Id1);

    EXPECT_EQ(message.secondSymmetryPositions().size(), 1);
    EXPECT_EQ(message.secondSymmetryPositions()[0].x(), X2);
    EXPECT_EQ(message.secondSymmetryPositions()[0].y(), Y2);
    EXPECT_EQ(message.secondSymmetryPositions()[0].type(), Type2);
    EXPECT_EQ(message.secondSymmetryPositions()[0].id(), Id2);
}

TEST(PositionConfirmationMessageTests, serialization_shouldSerializaToJson)
{
    constexpr double X1 = 10;
    constexpr double Y1 = 12;
    constexpr PositionType Type1 = PositionType::Speaker;
    constexpr uint32_t Id1 = 2;
    const vector<ConfigurationPosition> FirstSymmetryPositions{ ConfigurationPosition(X1, Y1, Type1, Id1) };

    constexpr double X2 = 9;
    constexpr double Y2 = 5;
    constexpr PositionType Type2 = PositionType::Probe;
    constexpr uint32_t Id2 = 3;
    const vector<ConfigurationPosition> SecondSymmetryPositions{ ConfigurationPosition(X2, Y2, Type2, Id2) };

    PositionConfirmationMessage message(FirstSymmetryPositions, SecondSymmetryPositions);
    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 3);

    EXPECT_EQ(serializedMessage.at("data").at("firstSymmetryPositions").size(), 1);
    EXPECT_EQ(serializedMessage.at("data").at("firstSymmetryPositions")[0].at("x"), X1);
    EXPECT_EQ(serializedMessage.at("data").at("firstSymmetryPositions")[0].at("y"), Y1);
    EXPECT_EQ(serializedMessage.at("data").at("firstSymmetryPositions")[0].at("type"), "s");
    EXPECT_EQ(serializedMessage.at("data").at("firstSymmetryPositions")[0].at("id"), Id1);

    EXPECT_EQ(serializedMessage.at("data").at("secondSymmetryPositions").size(), 1);
    EXPECT_EQ(serializedMessage.at("data").at("secondSymmetryPositions")[0].at("x"), X2);
    EXPECT_EQ(serializedMessage.at("data").at("secondSymmetryPositions")[0].at("y"), Y2);
    EXPECT_EQ(serializedMessage.at("data").at("secondSymmetryPositions")[0].at("type"), "m");
    EXPECT_EQ(serializedMessage.at("data").at("secondSymmetryPositions")[0].at("id"), Id2);

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
        "        \"type\": \"s\","
        "        \"id\": 2"
        "      }"
        "    ],"
        "    \"secondSymmetryPositions\": ["
        "      {"
        "        \"x\": 340,"
        "        \"y\": 140,"
        "        \"type\": \"m\","
        "        \"id\": 3"
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
    EXPECT_EQ(deserializedMessage.firstSymmetryPositions()[0].id(), 2);

    EXPECT_EQ(deserializedMessage.secondSymmetryPositions().size(), 1);
    EXPECT_EQ(deserializedMessage.secondSymmetryPositions()[0].x(), 340);
    EXPECT_EQ(deserializedMessage.secondSymmetryPositions()[0].y(), 140);
    EXPECT_EQ(deserializedMessage.secondSymmetryPositions()[0].type(), PositionType::Probe);
    EXPECT_EQ(deserializedMessage.secondSymmetryPositions()[0].id(), 3);
}
