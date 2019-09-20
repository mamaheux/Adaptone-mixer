#include <Uniformization/Communication/Messages/Udp/ProbeDiscoveryRequestMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(ProbeDiscoveryRequestMessageTests, constructor_shouldSetTheId)
{
    ProbeDiscoveryRequestMessage message;

    EXPECT_EQ(message.id(), 0);
    EXPECT_EQ(message.fullSize(), 4);
}

TEST(ProbeDiscoveryRequestMessageTests, fromBuffer_wrongId_shouldThrowInvalidValueException)
{
    constexpr size_t MessageSize = 4;
    uint8_t messageData[MessageSize] = { 0, 0, 0, 1 };
    NetworkBufferView buffer(messageData, MessageSize);

    EXPECT_THROW(ProbeDiscoveryRequestMessage::fromBuffer(buffer, MessageSize), InvalidValueException);
}

TEST(ProbeDiscoveryRequestMessageTests, fromBuffer_wrongMessageLength_shouldThrowInvalidValueException)
{
    constexpr size_t MessageSize = 4;
    uint8_t messageData[MessageSize] = { 0, 0, 0, 0 };
    NetworkBufferView buffer(messageData, MessageSize);

    EXPECT_THROW(ProbeDiscoveryRequestMessage::fromBuffer(buffer, MessageSize + 1), InvalidValueException);
}

TEST(ProbeDiscoveryRequestMessageTests, fromBuffer_shouldDeserialize)
{
    constexpr size_t MessageSize = 4;
    uint8_t messageData[MessageSize] = { 0, 0, 0, 0 };
    NetworkBufferView buffer(messageData, MessageSize);

    ProbeDiscoveryRequestMessage message = ProbeDiscoveryRequestMessage::fromBuffer(buffer, MessageSize);

    EXPECT_EQ(message.id(), 0);
    EXPECT_EQ(message.fullSize(), MessageSize);
}
