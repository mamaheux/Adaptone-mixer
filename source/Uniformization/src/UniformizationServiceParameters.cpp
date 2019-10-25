#include <Uniformization/UniformizationServiceParameters.h>

using namespace adaptone;
using namespace std;

UniformizationServiceParameters::UniformizationServiceParameters(Endpoint discoveryEndpoint,
    int discoveryTimeoutMs,
    size_t discoveryTrialCount,
    uint16_t tcpConnectionPort,
    uint16_t udpReceivingPort,
    int probeTimeoutMs,
    size_t sampleFrequency,
    double sweepDuration,
    double sweepMaxDelay,
    double outputHardwareDelay,
    double speedOfSound,
    double autoPositionAlpha,
    double autoPositionEpsilonTotalDistanceError,
    double autoPositionEpsilonDeltaTotalDistanceError,
    double autoPositionDistanceRelativeError,
    size_t autoPositionIterationCount,
    size_t autoPositionThermalIterationCount,
    size_t autoPositionTryCount,
    size_t autoPositionCountThreshold,
    vector<double> eqCenterFrequencies,
    PcmAudioFrameFormat format) :
    m_discoveryEndpoint(discoveryEndpoint),
    m_discoveryTimeoutMs(discoveryTimeoutMs),
    m_discoveryTrialCount(discoveryTrialCount),
    m_tcpConnectionPort(tcpConnectionPort),
    m_udpReceivingPort(udpReceivingPort),
    m_probeTimeoutMs(probeTimeoutMs),
    m_sampleFrequency(sampleFrequency),
    m_sweepDuration(sweepDuration),
    m_sweepMaxDelay(sweepMaxDelay),
    m_outputHardwareDelay(outputHardwareDelay),
    m_speedOfSound(speedOfSound),
    m_autoPositionAlpha(autoPositionAlpha),
    m_autoPositionEpsilonTotalDistanceError(autoPositionEpsilonTotalDistanceError),
    m_autoPositionEpsilonDeltaTotalDistanceError(autoPositionEpsilonDeltaTotalDistanceError),
    m_autoPositionDistanceRelativeError(autoPositionDistanceRelativeError),
    m_autoPositionIterationCount(autoPositionIterationCount),
    m_autoPositionThermalIterationCount(autoPositionThermalIterationCount),
    m_autoPositionTryCount(autoPositionTryCount),
    m_autoPositionCountThreshold(autoPositionCountThreshold),
    m_eqCenterFrequencies(eqCenterFrequencies),
    m_format(format)
{
}

UniformizationServiceParameters::~UniformizationServiceParameters()
{
}
