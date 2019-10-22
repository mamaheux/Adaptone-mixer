#include <Uniformization/UniformizationServiceParameters.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(UniformizationServiceParametersTests, constructor_shouldSetTheAttributes)
{
    const Endpoint DiscoveryEndpoint("192.168.1.1", 10);
    constexpr int DiscoveryTimeoutMs = 1;
    constexpr size_t DiscoveryTrialCount = 2;
    constexpr uint16_t TcpConnectionPort = 3;
    constexpr uint16_t UdpReceivingPort = 4;
    constexpr int ProbeTimeoutMs = 5;
    constexpr size_t SampleFrequency = 6;
    constexpr float SweepDuration = 7.7;
    constexpr float SweepMaxDelay = 8.8;
    constexpr PcmAudioFrameFormat Format = PcmAudioFrameFormat::Double;

    UniformizationServiceParameters parameters(DiscoveryEndpoint,
        DiscoveryTimeoutMs,
        DiscoveryTrialCount,
        TcpConnectionPort,
        UdpReceivingPort,
        ProbeTimeoutMs,
        SampleFrequency,
        SweepDuration,
        SweepMaxDelay,
        Format);

    EXPECT_EQ(parameters.discoveryEndpoint().ipAddress(), DiscoveryEndpoint.ipAddress());
    EXPECT_EQ(parameters.discoveryEndpoint().port(), DiscoveryEndpoint.port());
    EXPECT_EQ(parameters.discoveryTimeoutMs(), DiscoveryTimeoutMs);
    EXPECT_EQ(parameters.discoveryTrialCount(), DiscoveryTrialCount);
    EXPECT_EQ(parameters.tcpConnectionPort(), TcpConnectionPort);
    EXPECT_EQ(parameters.udpReceivingPort(), UdpReceivingPort);
    EXPECT_EQ(parameters.probeTimeoutMs(), ProbeTimeoutMs);
    EXPECT_EQ(parameters.sampleFrequency(), SampleFrequency);
    EXPECT_EQ(parameters.sweepDuration(), SweepDuration);
    EXPECT_EQ(parameters.sweepMaxDelay(), SweepMaxDelay);
    EXPECT_EQ(parameters.format(), Format);


    ProbeServerParameters probeServerParameters = parameters.toProbeServerParameters();
    EXPECT_EQ(probeServerParameters.discoveryEndpoint().ipAddress(), DiscoveryEndpoint.ipAddress());
    EXPECT_EQ(probeServerParameters.discoveryEndpoint().port(), DiscoveryEndpoint.port());
    EXPECT_EQ(probeServerParameters.discoveryTimeoutMs(), DiscoveryTimeoutMs);
    EXPECT_EQ(probeServerParameters.discoveryTrialCount(), DiscoveryTrialCount);
    EXPECT_EQ(probeServerParameters.tcpConnectionPort(), TcpConnectionPort);
    EXPECT_EQ(probeServerParameters.udpReceivingPort(), UdpReceivingPort);
    EXPECT_EQ(probeServerParameters.probeTimeoutMs(), ProbeTimeoutMs);
    EXPECT_EQ(probeServerParameters.sampleFrequency(), SampleFrequency);
    EXPECT_EQ(probeServerParameters.format(), Format);
}
