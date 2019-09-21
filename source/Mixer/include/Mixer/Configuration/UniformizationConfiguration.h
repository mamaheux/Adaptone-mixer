#ifndef MIXER_CONFIGURATION_UNIFORMIZATION_CONFIGURATION_H
#define MIXER_CONFIGURATION_UNIFORMIZATION_CONFIGURATION_H

#include <Utils/Configuration/Properties.h>
#include <Utils/Network/Endpoint.h>

namespace adaptone
{
    class UniformizationConfiguration
    {
        Endpoint m_discoveryEndpoint;
        int m_discoveryTimeoutMs;

        uint16_t m_tcpConnectionPort;
        uint16_t m_udpReceivingPort;
        int m_probeTimeoutMs;

    public:
        explicit UniformizationConfiguration(const Properties& properties);
        virtual ~UniformizationConfiguration();

        const Endpoint& discoveryEndpoint() const;
        int discoveryTimeoutMs() const;

        uint16_t tcpConnectionPort() const;
        uint16_t udpReceivingPort() const;
        int probeTimeoutMs() const;
    };

    inline const Endpoint& UniformizationConfiguration::discoveryEndpoint() const
    {
        return m_discoveryEndpoint;
    }

    inline int UniformizationConfiguration::discoveryTimeoutMs() const
    {
        return m_discoveryTimeoutMs;
    }

    inline uint16_t UniformizationConfiguration::tcpConnectionPort() const
    {
        return m_tcpConnectionPort;
    }

    inline uint16_t UniformizationConfiguration::udpReceivingPort() const
    {
        return m_udpReceivingPort;
    }

    inline int UniformizationConfiguration::probeTimeoutMs() const
    {
        return m_probeTimeoutMs;
    }
}

#endif
