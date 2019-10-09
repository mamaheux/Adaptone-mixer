#include <Uniformization/Communication/Messages/Tcp/ProbeInitializationResponseMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(ProbeInitializationResponseMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr bool IsCompatible = true;
    constexpr bool IsMaster = false;
    constexpr uint32_t ProbeId = 11;
    ProbeInitializationResponseMessage message(IsCompatible, IsMaster, ProbeId);

    EXPECT_EQ(message.id(), 3);
    EXPECT_EQ(message.fullSize(), 14);

    EXPECT_EQ(message.isCompatible(), IsCompatible);
    EXPECT_EQ(message.isMaster(), IsMaster);
    EXPECT_EQ(message.probeId(), ProbeId);
}

TEST(ProbeInitializationResponseMessageTests, toBuffer_shouldSerializeTheMessage)
{
    constexpr bool IsCompatible = true;
    constexpr bool IsMaster = false;
    constexpr uint32_t ProbeId = 11;
    ProbeInitializationResponseMessage message(IsCompatible, IsMaster, ProbeId);
    NetworkBuffer buffer(100);

    message.toBuffer(buffer);

    EXPECT_EQ(message.id(), 3);
    EXPECT_EQ(message.fullSize(), 14);

    EXPECT_EQ(buffer.data()[8], IsCompatible);
    EXPECT_EQ(buffer.data()[9], IsMaster);
    EXPECT_EQ(buffer.data()[10], 0);
    EXPECT_EQ(buffer.data()[11], 0);
    EXPECT_EQ(buffer.data()[12], 0);
    EXPECT_EQ(buffer.data()[13], ProbeId);
}

TEST(ProbeInitializationResponseMessageTests, fromBuffer_wrongId_shouldThrowInvalidValueException)
{
    constexpr size_t MessageSize = 14;
    uint8_t messageData[MessageSize] =
    {
        0, 0, 0, 2,
        0, 0, 0, 2,
        1, 0, 0, 0,
        0, 11
    };
    NetworkBufferView buffer(messageData, MessageSize);

    EXPECT_THROW(ProbeInitializationResponseMessage::fromBuffer(buffer, MessageSize), InvalidValueException);
}

TEST(ProbeInitializationResponseMessageTests, fromBuffer_wrongMessageLength_shouldThrowInvalidValueException)
{
    constexpr size_t MessageSize = 14;
    uint8_t messageData[MessageSize] =
    {
        0, 0, 0, 3,
        0, 0, 0, 2,
        1, 0, 0, 0,
        0, 11
    };
    NetworkBufferView buffer(messageData, MessageSize);

    EXPECT_THROW(ProbeInitializationResponseMessage::fromBuffer(buffer, MessageSize - 1), InvalidValueException);
}

TEST(ProbeInitializationResponseMessageTests, fromBuffer_shouldDeserialize)
{
    constexpr size_t MessageSize = 14;
    uint8_t messageData[MessageSize] =
    {
        0, 0, 0, 3,
        0, 0, 0, 2,
        1, 0, 0, 0,
        0, 11
    };
    NetworkBufferView buffer(messageData, MessageSize);

    ProbeInitializationResponseMessage message = ProbeInitializationResponseMessage::fromBuffer(buffer, MessageSize);

    EXPECT_EQ(message.id(), 3);
    EXPECT_EQ(message.fullSize(), MessageSize);

    EXPECT_TRUE(message.isCompatible());
    EXPECT_FALSE(message.isMaster());
    EXPECT_EQ(message.probeId(), 11);
}
