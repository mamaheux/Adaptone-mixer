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
        std::size_t m_discoveryTrialCount;

        uint16_t m_tcpConnectionPort;
        uint16_t m_udpReceivingPort;
        int m_probeTimeoutMs;

        double m_routineIRSweepF1;
        double m_routineIRSweepF2;
        double m_routineIRSweepT;
        double m_routineIRSweepMaxDelay;

        double m_speedOfSound;

        double m_autoPositionAlpha;
        double m_autoPositionEpsilonTotalDistanceError;
        double m_autoPositionEpsilonDeltaTotalDistanceError;
        double m_autoPositionDistanceRelativeError;
        std::size_t m_autoPositionIterationCount;
        std::size_t m_autoPositionThermalIterationCount;
        std::size_t m_autoPositionTryCount;
        std::size_t m_autoPositionCountThreshold;

    public:
        explicit UniformizationConfiguration(const Properties& properties);
        virtual ~UniformizationConfiguration();

        const Endpoint& discoveryEndpoint() const;
        int discoveryTimeoutMs() const;
        std::size_t discoveryTrialCount() const;

        uint16_t tcpConnectionPort() const;
        uint16_t udpReceivingPort() const;
        int probeTimeoutMs() const;

        double routineIRSweepF1() const;
        double routineIRSweepF2() const;
        double routineIRSweepT() const;
        double routineIRSweepMaxDelay() const;

        double speedOfSound() const;

        double autoPositionAlpha() const;
        double autoPositionEpsilonTotalDistanceError() const;
        double autoPositionEpsilonDeltaTotalDistanceError() const;
        double autoPositionDistanceRelativeError() const;
        std::size_t autoPositionIterationCount() const;
        std::size_t autoPositionThermalIterationCount() const;
        std::size_t autoPositionTryCount() const;
        std::size_t autoPositionCountThreshold() const;
    };

    inline const Endpoint& UniformizationConfiguration::discoveryEndpoint() const
    {
        return m_discoveryEndpoint;
    }

    inline int UniformizationConfiguration::discoveryTimeoutMs() const
    {
        return m_discoveryTimeoutMs;
    }

    inline std::size_t UniformizationConfiguration::discoveryTrialCount() const
    {
        return m_discoveryTrialCount;
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

    inline double UniformizationConfiguration::routineIRSweepF1() const
    {
        return m_routineIRSweepF1;
    }

    inline double UniformizationConfiguration::routineIRSweepF2() const
    {
        return m_routineIRSweepF2;
    }

    inline double UniformizationConfiguration::routineIRSweepT() const
    {
        return m_routineIRSweepT;
    }

    inline double UniformizationConfiguration::routineIRSweepMaxDelay() const
    {
        return m_routineIRSweepMaxDelay;
    }

    inline double UniformizationConfiguration::speedOfSound() const
    {
        return m_speedOfSound;
    }

    inline double UniformizationConfiguration::autoPositionAlpha() const
    {
        return m_autoPositionAlpha;
    }
    inline double UniformizationConfiguration::autoPositionEpsilonTotalDistanceError() const
    {
        return m_autoPositionEpsilonTotalDistanceError;
    }

    inline double UniformizationConfiguration::autoPositionEpsilonDeltaTotalDistanceError() const
    {
        return m_autoPositionEpsilonDeltaTotalDistanceError;
    }

    inline double UniformizationConfiguration::autoPositionDistanceRelativeError() const
    {
        return m_autoPositionDistanceRelativeError;
    }

    inline std::size_t UniformizationConfiguration::autoPositionIterationCount() const
    {
        return m_autoPositionIterationCount;
    }

    inline std::size_t UniformizationConfiguration::autoPositionThermalIterationCount() const
    {
        return m_autoPositionThermalIterationCount;
    }

    inline std::size_t UniformizationConfiguration::autoPositionTryCount() const
    {
        return m_autoPositionTryCount;
    }

    inline std::size_t UniformizationConfiguration::autoPositionCountThreshold() const
    {
        return m_autoPositionCountThreshold;
    }
}

#endif
