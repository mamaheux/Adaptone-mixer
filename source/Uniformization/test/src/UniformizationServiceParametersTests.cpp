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
    constexpr double SweepDuration = 7.7;
    constexpr double SweepMaxDelay = 8.8;
    constexpr double OutputHardwareDelay = 9.9;
    constexpr double SpeedOfSound = 10.10;
    constexpr double AutoPositionAlpha = 11.11;
    constexpr double AutoPositionEpsilonTotalDistanceError = 12.12;
    constexpr double AutoPositionEpsilonDeltaTotalDistanceError = 13.13;
    constexpr double AutoPositionDistanceRelativeError = 14.14;
    constexpr size_t AutoPositionIterationCount = 15;
    constexpr size_t AutoPositionThermalIterationCount = 16;
    constexpr size_t AutoPositionTryCount = 17;
    constexpr size_t AutoPositionCountThreshold = 18;
    const vector<double> EqCenterFrequencies({ 19, 19 });
    constexpr std::size_t EqControlBlockSize = 20;
    constexpr std::size_t EqControlErrorWindowSize = 21;
    constexpr double EqControlErrorCorrectionFactor = 22.22;
    constexpr double EqControlErrorCorrectionUpperBound = 23.23;
    constexpr double EqControlErrorCorrectionLowerBound = 24.24;
    constexpr double EqControlErrorCenterCorrectionFactor = 25.25;
    constexpr double EqControlEqGainUpperBoundDb = 26.26;
    constexpr double EqControlEqGainLowerBoundDb = 26.26;
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
        OutputHardwareDelay,
        SpeedOfSound,
        AutoPositionAlpha,
        AutoPositionEpsilonTotalDistanceError,
        AutoPositionEpsilonDeltaTotalDistanceError,
        AutoPositionDistanceRelativeError,
        AutoPositionIterationCount,
        AutoPositionThermalIterationCount,
        AutoPositionTryCount,
        AutoPositionCountThreshold,
        EqCenterFrequencies,
        EqControlBlockSize,
        EqControlErrorWindowSize,
        EqControlErrorCorrectionFactor,
        EqControlErrorCorrectionUpperBound,
        EqControlErrorCorrectionLowerBound,
        EqControlErrorCenterCorrectionFactor,
        EqControlEqGainUpperBoundDb,
        EqControlEqGainLowerBoundDb,
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
    EXPECT_DOUBLE_EQ(parameters.outputHardwareDelay(), OutputHardwareDelay);
    EXPECT_DOUBLE_EQ(parameters.speedOfSound(), SpeedOfSound);
    EXPECT_DOUBLE_EQ(parameters.autoPositionAlpha(), AutoPositionAlpha);
    EXPECT_DOUBLE_EQ(parameters.autoPositionEpsilonTotalDistanceError(), AutoPositionEpsilonTotalDistanceError);
    EXPECT_DOUBLE_EQ(parameters.autoPositionEpsilonDeltaTotalDistanceError(), AutoPositionEpsilonDeltaTotalDistanceError);
    EXPECT_DOUBLE_EQ(parameters.autoPositionDistanceRelativeError(), AutoPositionDistanceRelativeError);
    EXPECT_EQ(parameters.autoPositionIterationCount(), AutoPositionIterationCount);
    EXPECT_EQ(parameters.autoPositionThermalIterationCount(), AutoPositionThermalIterationCount);
    EXPECT_EQ(parameters.autoPositionTryCount(), AutoPositionTryCount);
    EXPECT_EQ(parameters.autoPositionCountThreshold(), AutoPositionCountThreshold);
    EXPECT_EQ(parameters.eqCenterFrequencies(), EqCenterFrequencies);
    EXPECT_EQ(parameters.eqControlBlockSize(), EqControlBlockSize);
    EXPECT_EQ(parameters.eqControlErrorWindowSize(), EqControlErrorWindowSize);
    EXPECT_DOUBLE_EQ(parameters.eqControlErrorCorrectionFactor(), EqControlErrorCorrectionFactor);
    EXPECT_DOUBLE_EQ(parameters.eqControlErrorCorrectionUpperBound(), EqControlErrorCorrectionUpperBound);
    EXPECT_DOUBLE_EQ(parameters.eqControlErrorCorrectionLowerBound(), EqControlErrorCorrectionLowerBound);
    EXPECT_DOUBLE_EQ(parameters.eqControlErrorCenterCorrectionFactor(), EqControlErrorCenterCorrectionFactor);
    EXPECT_DOUBLE_EQ(parameters.eqControlEqGainUpperBoundDb(), EqControlEqGainUpperBoundDb);
    EXPECT_DOUBLE_EQ(parameters.eqControlEqGainLowerBoundDb(), EqControlEqGainLowerBoundDb);
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
