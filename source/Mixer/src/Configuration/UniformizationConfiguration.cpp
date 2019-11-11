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

    constexpr const char* eqControlBlockSizePropertyKey = "uniformization.eq_control_block_size";
    constexpr const char* eqControlErrorWindowSizePropertyKey = "uniformization.eq_control_error_window_size";
    constexpr const char* eqControlErrorCorrectionFactorPropertyKey = "uniformization.eq_control_error_correction_factor";
    constexpr const char* eqControlErrorCorrectionUpperBoundPropertyKey = "uniformization.eq_control_error_correction_upper_bound";
    constexpr const char* eqControlErrorCorrectionLowerBoundPropertyKey = "uniformization.eq_control_error_correction_lower_bound";
    constexpr const char* eqControlErrorCenterCorrectionFactorPropertyKey = "uniformization.eq_control_error_center_correction_factor";
    constexpr const char* eqControlEqGainUpperBoundDbPropertyKey = "uniformization.eq_control_eq_gain_upper_bound_db";
    constexpr const char* eqControlEqGainLowerBoundDbPropertyKey = "uniformization.eq_control_eq_gain_lower_bound_db";

    m_discoveryEndpoint = properties.get<Endpoint>(DiscoveryEndpointPropertyKey);
    m_discoveryTimeoutMs = properties.get<int>(DiscoveryTimeoutMsPropertyKey);
    m_discoveryTrialCount = properties.get<size_t>(DiscoveryTrialCountPropertyKey);

    m_tcpConnectionPort = properties.get<uint16_t>(TcpConnectionPortPropertyKey);
    m_udpReceivingPort = properties.get<uint16_t>(UdpReceivingPortPropertyKey);
    m_probeTimeoutMs = properties.get<int>(ProbeTimeoutMsPropertyKey);

    m_routineIRSweepF1 = properties.get<double>(RoutineIRSweepF1PropertyKey);
    m_routineIRSweepF2 = properties.get<double>(RoutineIRSweepF2PropertyKey);
    m_routineIRSweepT = properties.get<double>(RoutineIRSweepTPropertyKey);
    m_routineIRSweepMaxDelay = properties.get<double>(RoutineIRSweepMaxDelayPropertyKey);

    m_speedOfSound = properties.get<double>(SpeedOfSoundPropertyKey);

    m_autoPositionAlpha = properties.get<double>(AutoPositionAlphaPropertyKey);
    m_autoPositionEpsilonTotalDistanceError = properties.get<double>(AutoPositionEpsilonTotalDistanceErrorPropertyKey);
    m_autoPositionEpsilonDeltaTotalDistanceError = properties.get<double>(AutoPositionEpsilonDeltaTotalDistanceErrorPropertyKey);
    m_autoPositionDistanceRelativeError = properties.get<double>(AutoPositionDistanceRelativeErrorPropertyKey);
    m_autoPositionIterationCount = properties.get<size_t>(AutoPositionIterationCountPropertyKey);
    m_autoPositionThermalIterationCount = properties.get<size_t>(AutoPositionThermalIterationCountPropertyKey);
    m_autoPositionTryCount = properties.get<size_t>(AutoPositionTryCountPropertyKey);
    m_autoPositionCountThreshold = properties.get<size_t>(AutoPositionCountThresholdPropertyKey);

    m_eqControlBlockSize = properties.get<size_t>(eqControlBlockSizePropertyKey);
    m_eqControlErrorWindowSize = properties.get<size_t>(eqControlErrorWindowSizePropertyKey);
    m_eqControlErrorCorrectionFactor = properties.get<double>(eqControlErrorCorrectionFactorPropertyKey);
    m_eqControlErrorCorrectionUpperBound = properties.get<double>(eqControlErrorCorrectionUpperBoundPropertyKey);
    m_eqControlErrorCorrectionLowerBound = properties.get<double>(eqControlErrorCorrectionLowerBoundPropertyKey);
    m_eqControlErrorCenterCorrectionFactor = properties.get<double>(eqControlErrorCenterCorrectionFactorPropertyKey);
    m_eqControlEqGainUpperBoundDb = properties.get<double>(eqControlEqGainUpperBoundDbPropertyKey);
    m_eqControlEqGainLowerBoundDb = properties.get<double>(eqControlEqGainLowerBoundDbPropertyKey);
}

UniformizationConfiguration::~UniformizationConfiguration()
{
}
