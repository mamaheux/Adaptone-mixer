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

        double m_autoPositionAlpha;
        double m_autoPositionEpsilonTotalDistanceError;
        double m_autoPositionEpsilonDeltaTotalDistanceError;
        double m_autoPositionDistanceRelativeError;
        std::size_t m_autoPositionIterationCount;
        std::size_t m_autoPositionThermalIterationCount;
        std::size_t m_autoPositionTryCount;
        std::size_t m_autoPositionCountThreshold;

        std::size_t m_eqControlBlockSize;
        std::size_t m_eqControlErrorWindowSize;
        double m_eqControlErrorCorrectionFactor;
        double m_eqControlErrorCorrectionUpperBound;
        double m_eqControlErrorCorrectionLowerBound;
        double m_eqControlErrorCenterCorrectionFactor;
        double m_eqControlEqGainUpperBoundDb;
        double m_eqControlEqGainLowerBoundDb;

        std::vector<double> m_eqCenterFrequencies;

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
            double autoPositionAlpha,
            double autoPositionEpsilonTotalDistanceError,
            double autoPositionEpsilonDeltaTotalDistanceError,
            double autoPositionDistanceRelativeError,
            std::size_t autoPositionIterationCount,
            std::size_t autoPositionThermalIterationCount,
            std::size_t autoPositionTryCount,
            std::size_t autoPositionCountThreshold,
            const std::vector<double>& eqCenterFrequencies,
            std::size_t eqControlBlockSize,
            std::size_t eqControlErrorWindowSize,
            double eqControlErrorCorrectionFactor,
            double eqControlErrorCorrectionUpperBound,
            double eqControlErrorCorrectionLowerBound,
            double eqControlErrorCenterCorrectionFactor,
            double eqControlEqGainUpperBoundDb,
            double eqControlEqGainLowerBoundDb,
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
        double autoPositionAlpha() const;
        double autoPositionEpsilonTotalDistanceError() const;
        double autoPositionEpsilonDeltaTotalDistanceError() const;
        double autoPositionDistanceRelativeError() const;
        std::size_t autoPositionIterationCount() const;
        std::size_t autoPositionThermalIterationCount() const;
        std::size_t autoPositionTryCount() const;
        std::size_t autoPositionCountThreshold() const;

        const std::vector<double>& eqCenterFrequencies() const;

        std::size_t eqControlBlockSize() const;
        std::size_t eqControlErrorWindowSize() const;
        double  eqControlErrorCorrectionFactor() const;
        double  eqControlErrorCorrectionUpperBound() const;
        double  eqControlErrorCorrectionLowerBound() const;
        double  eqControlErrorCenterCorrectionFactor() const;
        double  eqControlEqGainUpperBoundDb() const;
        double  eqControlEqGainLowerBoundDb() const;

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

    inline double UniformizationServiceParameters::autoPositionAlpha() const
    {
        return m_autoPositionAlpha;
    }

    inline double UniformizationServiceParameters::autoPositionEpsilonTotalDistanceError() const
    {
        return m_autoPositionEpsilonTotalDistanceError;
    }

    inline double UniformizationServiceParameters::autoPositionEpsilonDeltaTotalDistanceError() const
    {
        return m_autoPositionEpsilonDeltaTotalDistanceError;
    }

    inline double UniformizationServiceParameters::autoPositionDistanceRelativeError() const
    {
        return m_autoPositionDistanceRelativeError;
    }

    inline std::size_t UniformizationServiceParameters::autoPositionIterationCount() const
    {
        return m_autoPositionIterationCount;
    }

    inline std::size_t UniformizationServiceParameters::autoPositionThermalIterationCount() const
    {
        return m_autoPositionThermalIterationCount;
    }

    inline std::size_t UniformizationServiceParameters::autoPositionTryCount() const
    {
        return m_autoPositionTryCount;
    }

    inline std::size_t UniformizationServiceParameters::autoPositionCountThreshold() const
    {
        return m_autoPositionCountThreshold;
    }

    inline const std::vector<double>& UniformizationServiceParameters::eqCenterFrequencies() const
    {
        return m_eqCenterFrequencies;
    }

    inline std::size_t UniformizationServiceParameters::eqControlBlockSize() const
    {
        return m_eqControlBlockSize;
    }

    inline std::size_t UniformizationServiceParameters::eqControlErrorWindowSize() const
    {
        return m_eqControlErrorWindowSize;
    }

    inline double UniformizationServiceParameters::eqControlErrorCorrectionFactor() const
    {
        return m_eqControlErrorCorrectionFactor;
    }

    inline double UniformizationServiceParameters::eqControlErrorCorrectionUpperBound() const
    {
        return m_eqControlErrorCorrectionUpperBound;
    }

    inline double UniformizationServiceParameters::eqControlErrorCorrectionLowerBound() const
    {
        return m_eqControlErrorCorrectionLowerBound;
    }

    inline double UniformizationServiceParameters::eqControlErrorCenterCorrectionFactor() const
    {
        return m_eqControlErrorCenterCorrectionFactor;
    }

    inline double UniformizationServiceParameters::eqControlEqGainUpperBoundDb() const
    {
        return m_eqControlEqGainUpperBoundDb;
    }

    inline double UniformizationServiceParameters::eqControlEqGainLowerBoundDb() const
    {
        return m_eqControlEqGainLowerBoundDb;
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
