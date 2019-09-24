#include <Uniformization/Communication/Messages/Tcp/FftRequestMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(FftRequestMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr uint8_t Hours = 1;
    constexpr uint8_t Minutes = 2;
    constexpr uint8_t Seconds = 3;
    constexpr uint16_t Milliseconds = 4;
    constexpr uint16_t FftId = 5;
    FftRequestMessage message(Hours, Minutes, Seconds, Milliseconds, FftId);

    EXPECT_EQ(message.id(), 7);
    EXPECT_EQ(message.fullSize(), 15);

    EXPECT_EQ(message.hours(), Hours);
    EXPECT_EQ(message.minutes(), Minutes);
    EXPECT_EQ(message.seconds(), Seconds);
    EXPECT_EQ(message.milliseconds(), Milliseconds);
    EXPECT_EQ(message.fftId(), FftId);
}

TEST(FftRequestMessageTests, toBuffer_shouldSerializeTheMessage)
{
    constexpr uint8_t Hours = 1;
    constexpr uint8_t Minutes = 2;
    constexpr uint8_t Seconds = 3;
    constexpr uint16_t Milliseconds = 4;
    constexpr uint16_t FftId = 5;
    FftRequestMessage message(Hours, Minutes, Seconds, Milliseconds, FftId);
    NetworkBuffer buffer(100);

    message.toBuffer(buffer);

    EXPECT_EQ(message.id(), 7);
    EXPECT_EQ(message.fullSize(), 15);

    EXPECT_EQ(buffer.data()[8], 1);
    EXPECT_EQ(buffer.data()[9], 2);
    EXPECT_EQ(buffer.data()[10], 3);
    EXPECT_EQ(buffer.data()[11], 0);

    EXPECT_EQ(buffer.data()[12], 4);
    EXPECT_EQ(buffer.data()[13], 0);
    EXPECT_EQ(buffer.data()[14], 5);
}

TEST(FftRequestMessageTests, fromBuffer_wrongId_shouldThrowInvalidValueException)
{
    constexpr size_t MessageSize = 15;
    uint8_t messageData[MessageSize] =
    {
        0, 0, 0, 1,
        0, 0, 0, 7,
        1, 2, 3, 0,
        4, 0, 5
    };
    NetworkBufferView buffer(messageData, MessageSize);

    EXPECT_THROW(FftRequestMessage::fromBuffer(buffer, MessageSize), InvalidValueException);
}

TEST(FftRequestMessageTests, fromBuffer_wrongMessageLength_shouldThrowInvalidValueException)
{
    constexpr size_t MessageSize = 15;
    uint8_t messageData[MessageSize] =
    {
        0, 0, 0, 7,
        0, 0, 0, 7,
        1, 2, 3, 0,
        4, 0, 5
    };
    NetworkBufferView buffer(messageData, MessageSize);

    EXPECT_THROW(FftRequestMessage::fromBuffer(buffer, MessageSize + 1), InvalidValueException);
}

TEST(FftRequestMessageTests, fromBuffer_shouldDeserialize)
{
    constexpr size_t MessageSize = 15;
    uint8_t messageData[MessageSize] =
    {
        0, 0, 0, 7,
        0, 0, 0, 7,
        1, 2, 3, 0,
        4, 0, 5
    };
    NetworkBufferView buffer(messageData, MessageSize);

    FftRequestMessage message = FftRequestMessage::fromBuffer(buffer, MessageSize);

    EXPECT_EQ(message.id(), 7);
    EXPECT_EQ(message.fullSize(), MessageSize);

    EXPECT_EQ(message.hours(), 1);
    EXPECT_EQ(message.minutes(), 2);
    EXPECT_EQ(message.seconds(), 3);
    EXPECT_EQ(message.milliseconds(), 4);
    EXPECT_EQ(message.fftId(), 5);
}
