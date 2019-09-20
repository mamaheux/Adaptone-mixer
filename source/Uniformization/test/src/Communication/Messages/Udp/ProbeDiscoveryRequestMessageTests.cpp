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
