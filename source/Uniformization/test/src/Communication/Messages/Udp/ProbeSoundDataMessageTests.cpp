#include <Uniformization/Communication/Messages/Udp/ProbeSoundDataMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(ProbeSoundDataMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr uint16_t SoundDataId = 1;
    constexpr uint8_t Hours = 2;
    constexpr uint8_t Minutes = 3;
    constexpr uint8_t Seconds = 4;
    constexpr uint16_t Milliseconds = 5;
    constexpr uint16_t Microseconds = 6;
    constexpr size_t DataSize = 3;
    constexpr uint8_t Data[DataSize] = { 0, 1, 2 };
    ProbeSoundDataMessage message(SoundDataId, Hours, Minutes, Seconds, Milliseconds, Microseconds, Data, DataSize);

    EXPECT_EQ(message.id(), 9);
    EXPECT_EQ(message.fullSize(), 20);

    EXPECT_EQ(message.soundDataId(), SoundDataId);
    EXPECT_EQ(message.hours(), Hours);
    EXPECT_EQ(message.minutes(), Minutes);
    EXPECT_EQ(message.seconds(), Seconds);
    EXPECT_EQ(message.milliseconds(), Milliseconds);
    EXPECT_EQ(message.microseconds(), Microseconds);
    EXPECT_EQ(message.data(), Data);
    EXPECT_EQ(message.dataSize(), DataSize);
}

TEST(ProbeSoundDataMessageTests, toBuffer_shouldSerializeTheMessage)
{
    constexpr uint16_t SoundDataId = 1;
    constexpr uint8_t Hours = 2;
    constexpr uint8_t Minutes = 3;
    constexpr uint8_t Seconds = 4;
    constexpr uint16_t Milliseconds = 5;
    constexpr uint16_t Microseconds = 6;
    constexpr size_t DataSize = 3;
    constexpr uint8_t Data[DataSize] = { 0, 1, 2 };
    ProbeSoundDataMessage message(SoundDataId, Hours, Minutes, Seconds, Milliseconds, Microseconds, Data, DataSize);
    NetworkBuffer buffer(100);

    message.toBuffer(buffer);

    EXPECT_EQ(message.id(), 9);
    EXPECT_EQ(message.fullSize(), 20);

    EXPECT_EQ(buffer.data()[8], 0);
    EXPECT_EQ(buffer.data()[9], 1);
    EXPECT_EQ(buffer.data()[10], 2);
    EXPECT_EQ(buffer.data()[11], 3);

    EXPECT_EQ(buffer.data()[12], 4);
    EXPECT_EQ(buffer.data()[13], 0);
    EXPECT_EQ(buffer.data()[14], 5);
    EXPECT_EQ(buffer.data()[15], 0);

    EXPECT_EQ(buffer.data()[16], 6);
    EXPECT_EQ(buffer.data()[17], 0);
    EXPECT_EQ(buffer.data()[18], 1);
    EXPECT_EQ(buffer.data()[19], 2);
}
