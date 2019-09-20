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
