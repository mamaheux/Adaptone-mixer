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
