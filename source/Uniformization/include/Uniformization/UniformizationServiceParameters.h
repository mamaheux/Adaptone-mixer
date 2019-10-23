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
        double m_sweepDuration;
        double m_sweepMaxDelay;
        double m_outputHardwareDelay;
        double m_speedOfSound;
        PcmAudioFrameFormat m_format;

    public:
        UniformizationServiceParameters(Endpoint discoveryEndpoint,
            int discoveryTimeoutMs,
            std::size_t discoveryTrialCount,
            uint16_t tcpConnectionPort,
            uint16_t udpReceivingPort,
            int probeTimeoutMs,
            std::size_t sampleFrequency,
            double sweepDuration,
            double sweepMaxDelay,
            double outputHardwareDelay,
            double m_speedOfSound,
            PcmAudioFrameFormat format);
        virtual ~UniformizationServiceParameters();

        const Endpoint& discoveryEndpoint() const;
        int discoveryTimeoutMs() const;
        std::size_t discoveryTrialCount() const;
        uint16_t tcpConnectionPort() const;
        uint16_t udpReceivingPort() const;
        int probeTimeoutMs() const;
        std::size_t sampleFrequency() const;
        double sweepDuration() const;
        double sweepMaxDelay() const;
        double outputHardwareDelay() const;
        double speedOfSound() const;

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

    inline double UniformizationServiceParameters::sweepDuration() const
    {
        return m_sweepDuration;
    }

    inline double UniformizationServiceParameters::sweepMaxDelay() const
    {
        return m_sweepMaxDelay;
    }

    inline double UniformizationServiceParameters::outputHardwareDelay() const
    {
        return m_outputHardwareDelay;
    }

    inline double UniformizationServiceParameters::speedOfSound() const
    {
        return m_speedOfSound;
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
