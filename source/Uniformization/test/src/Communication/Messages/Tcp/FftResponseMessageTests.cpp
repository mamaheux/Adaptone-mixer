#include <Uniformization/Communication/Messages/Tcp/FftResponseMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(FftResponseMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr uint8_t FftId = 6;
    constexpr size_t FftValueCount = 3;
    constexpr complex<float> FftValues[FftValueCount] = { complex<float>(1, 2), complex<float>(2, 3), complex<float>(3, 4) };
    FftResponseMessage message(FftId, FftValues, FftValueCount);

    EXPECT_EQ(message.id(), 8);
    EXPECT_EQ(message.fullSize(), 34);

    EXPECT_EQ(message.fftId(), FftId);
    EXPECT_EQ(message.fftValues(), FftValues);
    EXPECT_EQ(message.fftValueCount(), FftValueCount);
}

TEST(FftResponseMessageTests, toBuffer_shouldSerializeTheMessage)
{
    constexpr uint8_t FftId = 6;
    constexpr size_t FftValueCount = 3;
    constexpr complex<float> FftValues[FftValueCount] = { complex<float>(1, 2), complex<float>(2, 3), complex<float>(3, 4) };
    FftResponseMessage message(FftId, FftValues, FftValueCount);
    NetworkBuffer buffer(100);

    message.toBuffer(buffer);

    EXPECT_EQ(message.id(), 8);
    EXPECT_EQ(message.fullSize(), 34);

    EXPECT_EQ(buffer.data()[8], 0);
    EXPECT_EQ(buffer.data()[9], 6);
    EXPECT_EQ(*reinterpret_cast<complex<float>*>(buffer.data() + 10), FftValues[0]);
    EXPECT_EQ(*reinterpret_cast<complex<float>*>(buffer.data() + 18), FftValues[1]);
    EXPECT_EQ(*reinterpret_cast<complex<float>*>(buffer.data() + 26), FftValues[2]);
}

TEST(FftResponseMessageTests, fromBuffer_wrongId_shouldThrowInvalidValueException)
{
    constexpr size_t MessageSize = 16;
    uint8_t messageData[MessageSize] =
    {
        0, 0, 0, 1,
        0, 0, 0, 8,
        1, 2, 3, 0,
        4, 0, 5, 6
    };
    NetworkBufferView buffer(messageData, MessageSize);

    EXPECT_THROW(FftResponseMessage::fromBuffer(buffer, MessageSize), InvalidValueException);
}

TEST(FftResponseMessageTests, fromBuffer_wrongMessageLength_shouldThrowInvalidValueException)
{
    constexpr size_t MessageSize = 10;
    uint8_t messageData[MessageSize] =
    {
        0, 0, 0, 8,
        0, 0, 0, 2,
        0, 1
    };
    NetworkBufferView buffer(messageData, MessageSize);

    EXPECT_THROW(FftResponseMessage::fromBuffer(buffer, MessageSize - 1), InvalidValueException);
}

TEST(FftResponseMessageTests, fromBuffer_shouldDeserialize)
{
    constexpr size_t MessageSize = 18;
    uint8_t messageData[MessageSize] =
    {
        0, 0, 0, 8,
        0, 0, 0, 10,
        0, 1, 0, 0,
        0, 0, 0, 0,
        0, 0
    };
    NetworkBufferView buffer(messageData, MessageSize);

    FftResponseMessage message = FftResponseMessage::fromBuffer(buffer, MessageSize);

    EXPECT_EQ(message.id(), 8);
    EXPECT_EQ(message.fullSize(), MessageSize);

    EXPECT_EQ(message.fftId(), 1);
    EXPECT_EQ(message.fftValues(), reinterpret_cast<complex<float>*>(messageData + 10));
    EXPECT_EQ(message.fftValueCount(), 1);
}
