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
