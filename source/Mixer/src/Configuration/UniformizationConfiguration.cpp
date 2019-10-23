#include <Mixer/Configuration/UniformizationConfiguration.h>

using namespace adaptone;

UniformizationConfiguration::UniformizationConfiguration(const Properties& properties)
{
    constexpr const char* DiscoveryEndpointPropertyKey = "uniformization.network.discovery_endpoint";
    constexpr const char* DiscoveryTimeoutMsPropertyKey = "uniformization.network.discovery_timeout_ms";
    constexpr const char* DiscoveryTrialCountPropertyKey = "uniformization.network.discovery_trial_count";

    constexpr const char* TcpConnectionPortPropertyKey = "uniformization.network.tcp_connection_port";
    constexpr const char* UdpReceivingPortPropertyKey = "uniformization.network.udp_receiving_port";
    constexpr const char* ProbeTimeoutMsPropertyKey = "uniformization.network.probe_timeout_ms";

    constexpr const char* RoutineIRSweepF1PropertyKey = "uniformization.routine_ir_sweep_f1";
    constexpr const char* RoutineIRSweepF2PropertyKey = "uniformization.routine_ir_sweep_f2";
    constexpr const char* RoutineIRSweepTPropertyKey = "uniformization.routine_ir_sweep_t";
    constexpr const char* RoutineIRSweepMaxDelayPropertyKey = "uniformization.routine_ir_sweep_max_delay";

    constexpr const char* SpeedOfSoundPropertyKey = "uniformization.speed_of_sound";

    constexpr const char* AutoPositionAlphaPropertyKey = "uniformization.auto_position_alpha";
    constexpr const char* AutoPositionEpsilonTotalDistanceErrorPropertyKey = "uniformization.auto_position_epsilon_total_distance_error";
    constexpr const char* AutoPositionEpsilonDeltaTotalDistanceErrorPropertyKey = "uniformization.auto_position_epsilon_delta_total_distance_error";
    constexpr const char* AutoPositionDistanceRelativeErrorPropertyKey = "uniformization.auto_position_distance_relative_error";
    constexpr const char* AutoPositionIterationCountPropertyKey = "uniformization.auto_position_iteration_count";
    constexpr const char* AutoPositionThermalIterationCountPropertyKey = "uniformization.auto_position_thermal_iteration_count";
    constexpr const char* AutoPositionTryCountPropertyKey = "uniformization.auto_position_try_count";
    constexpr const char* AutoPositionCountThresholdPropertyKey = "uniformization.auto_position_count_threshold";

    m_discoveryEndpoint = properties.get<Endpoint>(DiscoveryEndpointPropertyKey);
    m_discoveryTimeoutMs = properties.get<int>(DiscoveryTimeoutMsPropertyKey);
    m_discoveryTrialCount = properties.get<size_t>(DiscoveryTrialCountPropertyKey);

    m_tcpConnectionPort = properties.get<uint16_t>(TcpConnectionPortPropertyKey);
    m_udpReceivingPort = properties.get<uint16_t>(UdpReceivingPortPropertyKey);
    m_probeTimeoutMs = properties.get<int>(ProbeTimeoutMsPropertyKey);

    m_routineIRSweepF1 = properties.get<float>(RoutineIRSweepF1PropertyKey);
    m_routineIRSweepF2 = properties.get<float>(RoutineIRSweepF2PropertyKey);
    m_routineIRSweepT = properties.get<float>(RoutineIRSweepTPropertyKey);
    m_routineIRSweepMaxDelay = properties.get<float>(RoutineIRSweepMaxDelayPropertyKey);

    m_speedOfSound = properties.get<float>(SpeedOfSoundPropertyKey);

    m_autoPositionAlpha = properties.get<float>(AutoPositionAlphaPropertyKey);
    m_autoPositionEpsilonTotalDistanceError = properties.get<float>(AutoPositionEpsilonTotalDistanceErrorPropertyKey);
    m_autoPositionEpsilonDeltaTotalDistanceError = properties.get<float>(AutoPositionEpsilonDeltaTotalDistanceErrorPropertyKey);
    m_autoPositionDistanceRelativeError = properties.get<float>(AutoPositionDistanceRelativeErrorPropertyKey);
    m_autoPositionIterationCount = properties.get<size_t>(AutoPositionIterationCountPropertyKey);
    m_autoPositionThermalIterationCount = properties.get<size_t>(AutoPositionThermalIterationCountPropertyKey);
    m_autoPositionTryCount = properties.get<size_t>(AutoPositionTryCountPropertyKey);
    m_autoPositionCountThreshold = properties.get<size_t>(AutoPositionCountThresholdPropertyKey);
}

UniformizationConfiguration::~UniformizationConfiguration()
{
}
