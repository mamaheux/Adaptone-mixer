#include <Uniformization/Communication/Messages/Udp/ProbeDiscoveryResponseMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(ProbeDiscoveryResponseMessageTests, constructor_shouldSetTheId)
{
    ProbeDiscoveryResponseMessage message;

    EXPECT_EQ(message.id(), 1);
    EXPECT_EQ(message.fullSize(), 4);
}

TEST(ProbeDiscoveryResponseMessageTests, fromBuffer_wrongId_shouldThrowInvalidValueException)
{
    constexpr size_t MessageSize = 4;
    uint8_t messageData[MessageSize] = { 0, 0, 0, 0 };
    NetworkBufferView buffer(messageData, MessageSize);

    EXPECT_THROW(ProbeDiscoveryResponseMessage::fromBuffer(buffer, MessageSize), InvalidValueException);
}

TEST(ProbeDiscoveryResponseMessageTests, fromBuffer_wrongMessageSize_shouldThrowInvalidValueException)
{
    constexpr size_t MessageSize = 4;
    uint8_t messageData[MessageSize] = { 0, 0, 0, 1 };
    NetworkBufferView buffer(messageData, MessageSize);

    EXPECT_THROW(ProbeDiscoveryResponseMessage::fromBuffer(buffer, MessageSize + 1), InvalidValueException);
}

TEST(ProbeDiscoveryResponseMessageTests, fromBuffer_shouldDeserialize)
{
    constexpr size_t MessageSize = 4;
    uint8_t messageData[MessageSize] = { 0, 0, 0, 1 };
    NetworkBufferView buffer(messageData, MessageSize);

    ProbeDiscoveryResponseMessage message = ProbeDiscoveryResponseMessage::fromBuffer(buffer, MessageSize);

    EXPECT_EQ(message.id(), 1);
    EXPECT_EQ(message.fullSize(), MessageSize);
}
