#ifndef UNIFORMIZATION_UNIFORMIZATION_SERVICE_PARAMETERS_H
#define UNIFORMIZATION_UNIFORMIZATION_SERVICE_PARAMETERS_H

#include <Uniformization/Communication/ProbeServerParameters.h>

namespace adaptone
{
    class UniformizationServiceParameters
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
        UniformizationServiceParameters(Endpoint discoveryEndpoint,
            int discoveryTimeoutMs,
            std::size_t discoveryTrialCount,
            uint16_t tcpConnectionPort,
            uint16_t udpReceivingPort,
            int probeTimeoutMs,
            std::size_t sampleFrequency,
            PcmAudioFrameFormat format);
        virtual ~UniformizationServiceParameters();

        const Endpoint& discoveryEndpoint() const;
        int discoveryTimeoutMs() const;
        std::size_t discoveryTrialCount() const;
        uint16_t tcpConnectionPort() const;
        uint16_t udpReceivingPort() const;
        int probeTimeoutMs() const;
        std::size_t sampleFrequency() const;
        PcmAudioFrameFormat format() const;

        ProbeServerParameters toProbeServerParameters() const;
    };

    inline const Endpoint& UniformizationServiceParameters::discoveryEndpoint() const
    {
        return m_discoveryEndpoint;
    }

    inline int UniformizationServiceParameters::discoveryTimeoutMs() const
    {
        return m_discoveryTimeoutMs;
    }

    inline std::size_t UniformizationServiceParameters::discoveryTrialCount() const
    {
        return m_discoveryTrialCount;
    }

    inline uint16_t UniformizationServiceParameters::tcpConnectionPort() const
    {
        return m_tcpConnectionPort;
    }

    inline uint16_t UniformizationServiceParameters::udpReceivingPort() const
    {
        return m_udpReceivingPort;
    }

    inline int UniformizationServiceParameters::probeTimeoutMs() const
    {
        return m_probeTimeoutMs;
    }

    inline std::size_t UniformizationServiceParameters::sampleFrequency() const
    {
        return m_sampleFrequency;
    }

    inline PcmAudioFrameFormat UniformizationServiceParameters::format() const
    {
        return m_format;
    }

    inline ProbeServerParameters UniformizationServiceParameters::toProbeServerParameters() const
    {
        return ProbeServerParameters(m_discoveryEndpoint,
            m_discoveryTimeoutMs,
            m_discoveryTrialCount,
            m_tcpConnectionPort,
            m_udpReceivingPort,
            m_probeTimeoutMs,
            m_sampleFrequency,
            m_format);
    }
}

#endif
