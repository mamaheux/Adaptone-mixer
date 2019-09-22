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


    m_discoveryEndpoint = properties.get<Endpoint>(DiscoveryEndpointPropertyKey);
    m_discoveryTimeoutMs = properties.get<int>(DiscoveryTimeoutMsPropertyKey);
    m_discoveryTrialCount = properties.get<size_t>(DiscoveryTrialCountPropertyKey);

    m_tcpConnectionPort = properties.get<uint16_t>(TcpConnectionPortPropertyKey);
    m_udpReceivingPort = properties.get<uint16_t>(UdpReceivingPortPropertyKey);
    m_probeTimeoutMs = properties.get<int>(ProbeTimeoutMsPropertyKey);
}

UniformizationConfiguration::~UniformizationConfiguration()
{
}
