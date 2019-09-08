#include <Communication/Messages/Output/InputSpectrumMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(InputSpectrumMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr size_t ChannelId = 5;
    constexpr double Frequency = 1;
    constexpr double Amplitude = 2;
    const vector<ChannelSpectrum> channelSpectrums{
        ChannelSpectrum(ChannelId, { SpectrumPoint(Frequency, Amplitude) }) };
    InputSpectrumMessage message(channelSpectrums);

    EXPECT_EQ(message.seqId(), 20);

    ASSERT_EQ(message.channelSpectrums().size(), 1);
    EXPECT_EQ(message.channelSpectrums()[0].channelId(), ChannelId);
    ASSERT_EQ(message.channelSpectrums()[0].points().size(), 1);
    EXPECT_EQ(message.channelSpectrums()[0].points()[0].frequency(), Frequency);
    EXPECT_EQ(message.channelSpectrums()[0].points()[0].amplitude(), Amplitude);
}

TEST(InputSpectrumMessageTests, serialization_shouldSerializaToJson)
{
    constexpr size_t ChannelId = 5;
    constexpr double Frequency = 1;
    constexpr double Amplitude = 2;
    const vector<ChannelSpectrum> channelSpectrums{
        ChannelSpectrum(ChannelId, { SpectrumPoint(Frequency, Amplitude) }) };
    InputSpectrumMessage message(channelSpectrums);

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 20);

    ASSERT_EQ(serializedMessage.at("data").at("spectrums").size(), 1);
    EXPECT_EQ(serializedMessage.at("data").at("spectrums")[0].at("channelId"), ChannelId);
    ASSERT_EQ(serializedMessage.at("data").at("spectrums")[0].at("points").size(), 1);
    EXPECT_EQ(serializedMessage.at("data").at("spectrums")[0].at("points")[0].at("freq"), Frequency);
    EXPECT_EQ(serializedMessage.at("data").at("spectrums")[0].at("points")[0].at("amplitude"), Amplitude);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(InputSpectrumMessageTests, deserialization_shouldDeserializeFromJson)
{
    constexpr size_t ChannelId = 5;
    constexpr double Frequency = 1;
    constexpr double Amplitude = 2;
    string serializedMessage = "{"
        "  \"seqId\": 20,"
        "  \"data\": {"
        "    \"spectrums\": ["
        "      {"
        "        \"channelId\": 5,"
        "        \"points\": ["
        "          {"
        "            \"freq\": 1,"
        "            \"amplitude\": 2"
        "          }"
        "        ]"
        "      }"
        "    ]"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<InputSpectrumMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 20);

    ASSERT_EQ(deserializedMessage.channelSpectrums().size(), 1);
    EXPECT_EQ(deserializedMessage.channelSpectrums()[0].channelId(), ChannelId);
    ASSERT_EQ(deserializedMessage.channelSpectrums()[0].points().size(), 1);
    EXPECT_EQ(deserializedMessage.channelSpectrums()[0].points()[0].frequency(), Frequency);
    EXPECT_EQ(deserializedMessage.channelSpectrums()[0].points()[0].amplitude(), Amplitude);
}
