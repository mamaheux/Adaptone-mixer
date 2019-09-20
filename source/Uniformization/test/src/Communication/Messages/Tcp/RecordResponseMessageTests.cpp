#include <Uniformization/Communication/Messages/Tcp/RecordResponseMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(RecordResponseMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr uint8_t RecordId = 6;
    constexpr size_t DataSize = 3;
    constexpr uint8_t Data[DataSize] = { 0, 1, 2 };
    RecordResponseMessage message(RecordId, Data, DataSize);

    EXPECT_EQ(message.id(), 6);
    EXPECT_EQ(message.fullSize(), 12);

    EXPECT_EQ(message.recordId(), RecordId);
    EXPECT_EQ(message.data(), Data);
    EXPECT_EQ(message.dataSize(), DataSize);
}

TEST(RecordResponseMessageTests, toBuffer_shouldSerializeTheMessage)
{
    constexpr uint8_t RecordId = 6;
    constexpr size_t DataSize = 3;
    constexpr uint8_t Data[DataSize] = { 0, 1, 2 };
    RecordResponseMessage message(RecordId, Data, DataSize);
    NetworkBuffer buffer(100);

    message.toBuffer(buffer);

    EXPECT_EQ(message.id(), 6);
    EXPECT_EQ(message.fullSize(), 12);

    EXPECT_EQ(buffer.data()[8], RecordId);
    EXPECT_EQ(buffer.data()[9], Data[0]);
    EXPECT_EQ(buffer.data()[10], Data[1]);
    EXPECT_EQ(buffer.data()[11], Data[2]);
}

TEST(RecordResponseMessageTests, fromBuffer_wrongId_shouldThrowInvalidValueException)
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

    EXPECT_THROW(RecordResponseMessage::fromBuffer(buffer, MessageSize), InvalidValueException);
}

TEST(RecordResponseMessageTests, fromBuffer_wrongMessageLength_shouldThrowInvalidValueException)
{
    constexpr size_t MessageSize = 9;
    uint8_t messageData[MessageSize] =
    {
        0, 0, 0, 6,
        0, 0, 0, 1,
        1
    };
    NetworkBufferView buffer(messageData, MessageSize);

    EXPECT_THROW(RecordResponseMessage::fromBuffer(buffer, MessageSize - 1), InvalidValueException);
}

TEST(RecordResponseMessageTests, fromBuffer_shouldDeserialize)
{
    constexpr size_t MessageSize = 12;
    uint8_t messageData[MessageSize] =
    {
        0, 0, 0, 6,
        0, 0, 0, 4,
        1, 2, 3, 0
    };
    NetworkBufferView buffer(messageData, MessageSize);

    RecordResponseMessage message = RecordResponseMessage::fromBuffer(buffer, MessageSize);

    EXPECT_EQ(message.id(), 6);
    EXPECT_EQ(message.fullSize(), MessageSize);

    EXPECT_EQ(message.recordId(), 1);
    EXPECT_EQ(message.data(), messageData + 9);
    EXPECT_EQ(message.dataSize(), 3);
}
