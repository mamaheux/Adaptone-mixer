#include <Mixer/Configuration/UniformizationConfiguration.h>

#include <gtest/gtest.h>

using namespace adaptone;

TEST(UniformizationConfigurationTests, constructor_shouldSetTheAttributes)
{
    UniformizationConfiguration configuration(Properties(
    {
        { "uniformization.network.discovery_endpoint", "192.168.1.255:5000" },
        { "uniformization.network.discovery_timeout_ms", "1000" },
        { "uniformization.network.discovery_trial_count", "5" },

        { "uniformization.network.tcp_connection_port", "5001"},
        { "uniformization.network.udp_receiving_port", "5002"},
        { "uniformization.network.probe_timeout_ms", "2000"}
    }));

    EXPECT_EQ(configuration.discoveryEndpoint().ipAddress(), "192.168.1.255");
    EXPECT_EQ(configuration.discoveryEndpoint().port(), 5000);
    EXPECT_EQ(configuration.discoveryTimeoutMs(), 1000);
    EXPECT_EQ(configuration.discoveryTrialCount(), 5);

    EXPECT_EQ(configuration.tcpConnectionPort(), 5001);
    EXPECT_EQ(configuration.udpReceivingPort(), 5002);
    EXPECT_EQ(configuration.probeTimeoutMs(), 2000);
}
