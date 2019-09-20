#include <Uniformization/Communication/Messages/Tcp/HeartbeatMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(HeartbeatMessageTests, constructor_shouldSetTheId)
{
    HeartbeatMessage message;

    EXPECT_EQ(message.id(), 4);
    EXPECT_EQ(message.fullSize(), 4);
}

TEST(HeartbeatMessageTests, fromBuffer_wrongId_shouldThrowInvalidValueException)
{
    constexpr size_t MessageSize = 4;
    uint8_t messageData[MessageSize] = { 0, 0, 0, 1 };
    NetworkBufferView buffer(messageData, MessageSize);

    EXPECT_THROW(HeartbeatMessage::fromBuffer(buffer, MessageSize), InvalidValueException);
}

TEST(HeartbeatMessageTests, fromBuffer_wrongMessageLength_shouldThrowInvalidValueException)
{
    constexpr size_t MessageSize = 4;
    uint8_t messageData[MessageSize] = { 0, 0, 0, 4 };
    NetworkBufferView buffer(messageData, MessageSize);

    EXPECT_THROW(HeartbeatMessage::fromBuffer(buffer, MessageSize + 1), InvalidValueException);
}

TEST(HeartbeatMessageTests, fromBuffer_shouldDeserialize)
{
    constexpr size_t MessageSize = 4;
    uint8_t messageData[MessageSize] = { 0, 0, 0, 4 };
    NetworkBufferView buffer(messageData, MessageSize);

    HeartbeatMessage message = HeartbeatMessage::fromBuffer(buffer, MessageSize);

    EXPECT_EQ(message.id(), 4);
    EXPECT_EQ(message.fullSize(), MessageSize);
}
