#include <Uniformization/Communication/Messages/Tcp/ProbeInitializationResponseMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(ProbeInitializationResponseMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr bool IsCompatible = true;
    constexpr bool IsMaster = false;
    ProbeInitializationResponseMessage message(IsCompatible, IsMaster);

    EXPECT_EQ(message.id(), 3);
    EXPECT_EQ(message.fullSize(), 10);

    EXPECT_EQ(message.isCompatible(), IsCompatible);
    EXPECT_EQ(message.isMaster(), IsMaster);
}

TEST(ProbeInitializationResponseMessageTests, toBuffer_shouldSerializeTheMessage)
{
    constexpr bool IsCompatible = true;
    constexpr bool IsMaster = false;
    ProbeInitializationResponseMessage message(IsCompatible, IsMaster);
    NetworkBuffer buffer(100);

    message.toBuffer(buffer);

    EXPECT_EQ(message.id(), 3);
    EXPECT_EQ(message.fullSize(), 10);

    EXPECT_EQ(buffer.data()[8], IsCompatible);
    EXPECT_EQ(buffer.data()[9], IsMaster);
}

TEST(ProbeInitializationResponseMessageTests, fromBuffer_wrongId_shouldThrowInvalidValueException)
{
    constexpr size_t MessageSize = 10;
    uint8_t messageData[MessageSize] =
    {
        0, 0, 0, 2,
        0, 0, 0, 2,
        1, 0
    };
    NetworkBufferView buffer(messageData, MessageSize);

    EXPECT_THROW(ProbeInitializationResponseMessage::fromBuffer(buffer, MessageSize), InvalidValueException);
}

TEST(ProbeInitializationResponseMessageTests, fromBuffer_wrongMessageLength_shouldThrowInvalidValueException)
{
    constexpr size_t MessageSize = 10;
    uint8_t messageData[MessageSize] =
    {
        0, 0, 0, 3,
        0, 0, 0, 2,
        1, 0
    };
    NetworkBufferView buffer(messageData, MessageSize);

    EXPECT_THROW(ProbeInitializationResponseMessage::fromBuffer(buffer, MessageSize - 1), InvalidValueException);
}

TEST(ProbeInitializationResponseMessageTests, fromBuffer_shouldDeserialize)
{
    constexpr size_t MessageSize = 10;
    uint8_t messageData[MessageSize] =
    {
        0, 0, 0, 3,
        0, 0, 0, 2,
        1, 0
    };
    NetworkBufferView buffer(messageData, MessageSize);

    ProbeInitializationResponseMessage message = ProbeInitializationResponseMessage::fromBuffer(buffer, MessageSize);

    EXPECT_EQ(message.id(), 3);
    EXPECT_EQ(message.fullSize(), MessageSize);

    EXPECT_TRUE(message.isCompatible());
    EXPECT_FALSE(message.isMaster());
}
