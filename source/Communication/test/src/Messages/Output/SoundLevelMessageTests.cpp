#include <Communication/Messages/Output/SoundLevelMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

TEST(SoundLevelMessageTests, constructor_shouldSetTheAttributes)
{
    const vector<ChannelSoundLevel> InputAfterGain{ ChannelSoundLevel(0, 1), ChannelSoundLevel(1, 2) };
    const vector<ChannelSoundLevel> InputAfterEq{ ChannelSoundLevel(0, 4), ChannelSoundLevel(1, 4) };
    const vector<ChannelSoundLevel> OutputAfterGain{ ChannelSoundLevel(0, 5), ChannelSoundLevel(1, 6) };
    SoundLevelMessage message(InputAfterGain, InputAfterEq, OutputAfterGain);

    EXPECT_EQ(message.seqId(), 23);

    EXPECT_EQ(message.inputAfterGain(), InputAfterGain);
    EXPECT_EQ(message.inputAfterEq(), InputAfterEq);
    EXPECT_EQ(message.outputAfterGain(), OutputAfterGain);
}

TEST(SoundLevelMessageTests, serialization_shouldSerializaToJson)
{
    const vector<ChannelSoundLevel> InputAfterGain{ ChannelSoundLevel(0, 1), ChannelSoundLevel(1, 2) };
    const vector<ChannelSoundLevel> InputAfterEq{ ChannelSoundLevel(0, 4), ChannelSoundLevel(1, 4) };
    const vector<ChannelSoundLevel> OutputAfterGain{ ChannelSoundLevel(0, 5), ChannelSoundLevel(1, 6) };
    SoundLevelMessage message(InputAfterGain, InputAfterEq, OutputAfterGain);

    json serializedMessage = message;

    EXPECT_EQ(serializedMessage.at("seqId"), 23);

    EXPECT_EQ(serializedMessage.at("data").at("inputAfterGain"), InputAfterGain);
    EXPECT_EQ(serializedMessage.at("data").at("inputAfterEq"), InputAfterEq);
    EXPECT_EQ(serializedMessage.at("data").at("outputAfterGain"), OutputAfterGain);

    EXPECT_EQ(serializedMessage.dump(), message.toJson());
}

TEST(SoundLevelMessageTests, deserialization_shouldDeserializeFromJson)
{
    string serializedMessage = "{"
        "  \"seqId\": 23,"
        "  \"data\": {"
        "    \"inputAfterGain\": ["
        "      {"
        "        \"channelId\": 0,"
        "        \"level\": 1"
        "      },"
        "      {"
        "        \"channelId\": 1,"
        "        \"level\": 2"
        "      }"
        "    ],"
        "    \"inputAfterEq\": ["
        "      {"
        "        \"channelId\": 0,"
        "        \"level\": 3"
        "      },"
        "      {"
        "        \"channelId\": 1,"
        "        \"level\": 4"
        "      }"
        "    ],"
        "    \"outputAfterGain\": ["
        "      {"
        "        \"channelId\": 0,"
        "        \"level\": 5"
        "      },"
        "      {"
        "        \"channelId\": 1,"
        "        \"level\": 6"
        "      }"
        "    ]"
        "  }"
        "}";

    auto deserializedMessage = json::parse(serializedMessage).get<SoundLevelMessage>();

    EXPECT_EQ(deserializedMessage.seqId(), 23);

    EXPECT_EQ(deserializedMessage.inputAfterGain(),
        vector<ChannelSoundLevel>({ ChannelSoundLevel(0, 1), ChannelSoundLevel(1, 2) }));
    EXPECT_EQ(deserializedMessage.inputAfterEq(),
        vector<ChannelSoundLevel>({ ChannelSoundLevel(0, 3), ChannelSoundLevel(1, 4) }));
    EXPECT_EQ(deserializedMessage.outputAfterGain(),
        vector<ChannelSoundLevel>({ ChannelSoundLevel(0, 5), ChannelSoundLevel(1, 6) }));
}
