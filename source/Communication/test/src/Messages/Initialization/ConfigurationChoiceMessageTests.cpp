#include <Communication/Messages/Initialization/ConfigurationChoiceMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(ConfigurationChoiceMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr size_t Id = 10;
    const string Name = "a name";

    const vector<size_t> InputChannelIds{ 1, 2, 3 };
    constexpr size_t SpeakersNumber = 2;
    const vector<size_t> AuxiliaryChannelIds{ 4, 5 };

    constexpr double X = 10;
    constexpr double Y = 12;
    constexpr PositionType Type = PositionType::Speaker;
    const vector<ConfigurationPosition> Positions{ ConfigurationPosition(X, Y, Type) };

    ConfigurationChoiceMessage message(Id, Name, InputChannelIds, SpeakersNumber, AuxiliaryChannelIds, Positions);

    EXPECT_EQ(message.seqId(), 0);

    EXPECT_EQ(message.id(), Id);
    EXPECT_EQ(message.name(), Name);

    EXPECT_EQ(message.inputChannelIds(), vector<size_t>({ 1, 2, 3 }));
    EXPECT_EQ(message.speakersNumber(), SpeakersNumber);
    EXPECT_EQ(message.auxiliaryChannelIds(), vector<size_t>({ 4, 5 }));

    EXPECT_EQ(message.positions().size(), 1);
    EXPECT_EQ(message.positions()[0].x(), X);
    EXPECT_EQ(message.positions()[0].y(), Y);
    EXPECT_EQ(message.positions()[0].type(), Type);
}

TEST(ConfigurationChoiceMessageTests, serialization_shouldSerializaToJson)
{
    constexpr size_t Id = 10;
    const string Name = "a name";

    const vector<size_t> InputChannelIds{ 1, 2, 3 };
    constexpr size_t SpeakersNumber = 2;
    const vector<size_t> AuxiliaryChannelIds{ 4, 5 };

    constexpr double X = 10;
    constexpr double Y = 12;
    constexpr PositionType Type = PositionType::Speaker;
    const vector<ConfigurationPosition> Positions{ ConfigurationPosition(X, Y, Type) };

    ConfigurationChoiceMessage message(Id, Name, InputChannelIds, SpeakersNumber, AuxiliaryChannelIds, Positions);
    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 0);

    EXPECT_EQ(serializedMessage.at("data").at("id"), Id);
    EXPECT_EQ(serializedMessage.at("data").at("name"), Name);

    EXPECT_EQ(serializedMessage.at("data").at("inputChannelIds"), InputChannelIds);
    EXPECT_EQ(serializedMessage.at("data").at("speakersNumber"), SpeakersNumber);
    EXPECT_EQ(serializedMessage.at("data").at("auxiliaryChannelIds"), AuxiliaryChannelIds);

    EXPECT_EQ(serializedMessage.at("data").at("positions").size(), 1);
    EXPECT_EQ(serializedMessage.at("data").at("positions")[0].at("x"), X);
    EXPECT_EQ(serializedMessage.at("data").at("positions")[0].at("y"), Y);
    EXPECT_EQ(serializedMessage.at("data").at("positions")[0].at("type"), "s");

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(ConfigurationChoiceMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 0,"
        "  \"data\": {"
        "    \"id\": 10,"
        "    \"name\": \"super nom\","
        "    \"inputChannelIds\": [1, 2, 3],"
        "    \"speakersNumber\": 4,"
        "    \"auxiliaryChannelIds\": [4, 5],"
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

    EXPECT_EQ(deserializedMessage.inputChannelIds(), vector<size_t>({ 1, 2, 3 }));
    EXPECT_EQ(deserializedMessage.speakersNumber(), 4);
    EXPECT_EQ(deserializedMessage.auxiliaryChannelIds(), vector<size_t>({ 4, 5 }));

    EXPECT_EQ(deserializedMessage.positions().size(), 1);
    EXPECT_EQ(deserializedMessage.positions()[0].x(), 140);
    EXPECT_EQ(deserializedMessage.positions()[0].y(), 340);
    EXPECT_EQ(deserializedMessage.positions()[0].type(), PositionType::Speaker);
}
