#include <Uniformization/Communication/Messages/Tcp/RecordRequestMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(RecordRequestMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr uint8_t Hours = 1;
    constexpr uint8_t Minutes = 2;
    constexpr uint8_t Seconds = 3;
    constexpr uint16_t Milliseconds = 4;
    constexpr uint16_t Duration = 5;
    constexpr uint8_t RecordId = 6;
    RecordRequestMessage message(Hours, Minutes, Seconds, Milliseconds, Duration, RecordId);

    EXPECT_EQ(message.id(), 5);
    EXPECT_EQ(message.fullSize(), 16);

    EXPECT_EQ(message.hours(), Hours);
    EXPECT_EQ(message.minutes(), Minutes);
    EXPECT_EQ(message.seconds(), Seconds);
    EXPECT_EQ(message.milliseconds(), Milliseconds);
    EXPECT_EQ(message.duration(), Duration);
    EXPECT_EQ(message.recordId(), RecordId);
}

TEST(RecordRequestMessageTests, toBuffer_shouldSerializeTheMessage)
{
    constexpr uint8_t Hours = 1;
    constexpr uint8_t Minutes = 2;
    constexpr uint8_t Seconds = 3;
    constexpr uint16_t Milliseconds = 4;
    constexpr uint16_t Duration = 5;
    constexpr uint8_t RecordId = 6;
    RecordRequestMessage message(Hours, Minutes, Seconds, Milliseconds, Duration, RecordId);
    NetworkBuffer buffer(100);

    message.toBuffer(buffer);

    EXPECT_EQ(message.id(), 5);
    EXPECT_EQ(message.fullSize(), 16);

    EXPECT_EQ(buffer.data()[8], 1);
    EXPECT_EQ(buffer.data()[9], 2);
    EXPECT_EQ(buffer.data()[10], 3);
    EXPECT_EQ(buffer.data()[11], 0);

    EXPECT_EQ(buffer.data()[12], 4);
    EXPECT_EQ(buffer.data()[13], 0);
    EXPECT_EQ(buffer.data()[14], 5);
    EXPECT_EQ(buffer.data()[15], 6);
}
