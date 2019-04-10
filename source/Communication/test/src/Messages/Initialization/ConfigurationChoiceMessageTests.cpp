#include <Communication/Messages/Initialization/ConfigurationChoiceMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ConfigurationChoiceMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr size_t id = 10;
    const string name = "a name";

    constexpr size_t monitorsNumber = 1;
    constexpr size_t speakersNumber = 2;
    constexpr size_t probesNumber = 3;

    constexpr double x = 10;
    constexpr double y = 12;
    constexpr ConfigurationPosition::Type type = ConfigurationPosition::Type::Speaker;
    const vector<ConfigurationPosition> positions{ ConfigurationPosition(x, y, type) };

    ConfigurationChoiceMessage message(id, name, monitorsNumber, speakersNumber, probesNumber, positions);

    EXPECT_EQ(message.seqId(), 0);

    EXPECT_EQ(message.id(), id);
    EXPECT_EQ(message.name(), name);

    EXPECT_EQ(message.monitorsNumber(), monitorsNumber);
    EXPECT_EQ(message.speakersNumber(), speakersNumber);
    EXPECT_EQ(message.probesNumber(), probesNumber);

    EXPECT_EQ(message.positions().size(), 1);
    EXPECT_EQ(message.positions()[0].x(), x);
    EXPECT_EQ(message.positions()[0].y(), y);
    EXPECT_EQ(message.positions()[0].type(), type);
}

TEST(ConfigurationChoiceMessageTests, serialization_shouldSerializaToJson)
{
    constexpr size_t id = 10;
    const string name = "a name";

    constexpr size_t monitorsNumber = 1;
    constexpr size_t speakersNumber = 2;
    constexpr size_t probesNumber = 3;

    constexpr double x = 10;
    constexpr double y = 12;
    constexpr ConfigurationPosition::Type type = ConfigurationPosition::Type::Speaker;
    const vector<ConfigurationPosition> positions{ ConfigurationPosition(x, y, type) };

    ConfigurationChoiceMessage message(id, name, monitorsNumber, speakersNumber, probesNumber, positions);
    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 0);

    EXPECT_EQ(serializedMessage.at("data").at("id"), id);
    EXPECT_EQ(serializedMessage.at("data").at("name"), name);

    EXPECT_EQ(serializedMessage.at("data").at("monitorsNumber"), monitorsNumber);
    EXPECT_EQ(serializedMessage.at("data").at("speakersNumber"), speakersNumber);
    EXPECT_EQ(serializedMessage.at("data").at("probesNumber"), probesNumber);

    EXPECT_EQ(serializedMessage.at("data").at("positions").size(), 1);
    EXPECT_EQ(serializedMessage.at("data").at("positions")[0].at("x"), x);
    EXPECT_EQ(serializedMessage.at("data").at("positions")[0].at("y"), y);
    EXPECT_EQ(serializedMessage.at("data").at("positions")[0].at("type"), "s");
}

TEST(ConfigurationChoiceMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 0,"
        "  \"data\": {"
        "    \"id\": 10,"
        "    \"name\": \"super nom\","
        "    \"monitorsNumber\": 5,"
        "    \"speakersNumber\": 4,"
        "    \"probesNumber\": 8,"
        "    \"positions\": ["
        "      {"
        "        \"x\": 140,"
        "        \"y\": 340,"
        "        \"type\": \"s\""
        "      }"
        "    ]"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<ConfigurationChoiceMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 0);

    EXPECT_EQ(deserializedMessage.id(), 10);
    EXPECT_EQ(deserializedMessage.name(), "super nom");

    EXPECT_EQ(deserializedMessage.monitorsNumber(), 5);
    EXPECT_EQ(deserializedMessage.speakersNumber(), 4);
    EXPECT_EQ(deserializedMessage.probesNumber(), 8);

    EXPECT_EQ(deserializedMessage.positions().size(), 1);
    EXPECT_EQ(deserializedMessage.positions()[0].x(), 140);
    EXPECT_EQ(deserializedMessage.positions()[0].y(), 340);
    EXPECT_EQ(deserializedMessage.positions()[0].type(), ConfigurationPosition::Type::Speaker);
}
