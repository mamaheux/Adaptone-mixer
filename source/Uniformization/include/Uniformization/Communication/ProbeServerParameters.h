#ifndef UNIFORMIZATION_COMMUNICATION_PROBE_SERVER_PARAMETERS_H
#define UNIFORMIZATION_COMMUNICATION_PROBE_SERVER_PARAMETERS_H

#include <Utils/Data/PcmAudioFrameFormat.h>
#include <Utils/Network/Endpoint.h>

#include <cstddef>
#include <cstdint>

namespace adaptone
{
    class ProbeServerParameters
    {
        Endpoint m_discoveryEndpoint;
        int m_discoveryTimeoutMs;
        std::size_t m_discoveryTrialCount;
        uint16_t m_tcpConnectionPort;
        uint16_t m_udpReceivingPort;
        int m_probeTimeoutMs;
        std::size_t m_sampleFrequency;
        PcmAudioFrameFormat m_format;

    public:
        ProbeServerParameters(Endpoint discoveryEndpoint,
            int discoveryTimeoutMs,
            std::size_t discoveryTrialCount,
            uint16_t tcpConnectionPort,
            uint16_t udpReceivingPort,
            int probeTimeoutMs,
            std::size_t sampleFrequency,
            PcmAudioFrameFormat format);
        virtual ~ProbeServerParameters();

        const Endpoint& discoveryEndpoint() const;
        int discoveryTimeoutMs() const;
        std::size_t discoveryTrialCount() const;
        uint16_t tcpConnectionPort() const;
        uint16_t udpReceivingPort() const;
        int probeTimeoutMs() const;
        std::size_t sampleFrequency() const;
        PcmAudioFrameFormat format() const;
    };

    inline const Endpoint& ProbeServerParameters::discoveryEndpoint() const
    {
        return m_discoveryEndpoint;
    }

    inline int ProbeServerParameters::discoveryTimeoutMs() const
    {
        return m_discoveryTimeoutMs;
    }

    inline std::size_t ProbeServerParameters::discoveryTrialCount() const
    {
        return m_discoveryTrialCount;
    }

    inline uint16_t ProbeServerParameters::tcpConnectionPort() const
    {
        return m_tcpConnectionPort;
    }

    inline uint16_t ProbeServerParameters::udpReceivingPort() const
    {
        return m_udpReceivingPort;
    }

    inline int ProbeServerParameters::probeTimeoutMs() const
    {
        return m_probeTimeoutMs;
    }

    inline std::size_t ProbeServerParameters::sampleFrequency() const
    {
        return m_sampleFrequency;
    }

    inline PcmAudioFrameFormat ProbeServerParameters::format() const
    {
        return m_format;
    }
}

#endif

