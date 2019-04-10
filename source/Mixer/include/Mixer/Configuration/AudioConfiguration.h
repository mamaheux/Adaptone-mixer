#ifndef MIXER_CONFIGURATION_AUDIO_CONFIGURATION_H
#define MIXER_CONFIGURATION_AUDIO_CONFIGURATION_H

#include <Utils/Configuration/Properties.h>
#include <SignalProcessing/ProcessingDataType.h>

#include <vector>

namespace adaptone
{
    class AudioConfiguration
    {
        std::size_t m_frameSampleCount;
        std::size_t m_sampleFrequency; // 44 kHz, 48 kHz, 96 kHz
        std::size_t m_inputChannelCount;
        std::size_t m_outputChannelCount;
        ProcessingDataType m_processingDataType;

        std::size_t m_parametricEqFilterCount;
        std::vector<double> m_eqCenterFrequencies;

        std::size_t m_soundLevelLength;

    public:
        explicit AudioConfiguration(const Properties& properties);
        virtual ~AudioConfiguration();

        std::size_t frameSampleCount() const;
        std::size_t sampleFrequency() const;
        std::size_t inputChannelCount() const;
        std::size_t outputChannelCount() const;
        ProcessingDataType processingDataType() const;

        std::size_t parametricEqFilterCount() const;
        const std::vector<double>& eqCenterFrequencies() const;

        std::size_t soundLevelLength() const;
    };

    inline std::size_t AudioConfiguration::frameSampleCount() const
    {
        return m_frameSampleCount;
    }

    inline std::size_t AudioConfiguration::sampleFrequency() const
    {
        return m_sampleFrequency;
    }

    inline std::size_t AudioConfiguration::inputChannelCount() const
    {
        return m_inputChannelCount;
    }

    inline std::size_t AudioConfiguration::outputChannelCount() const
    {
        return m_outputChannelCount;
    }

    inline ProcessingDataType AudioConfiguration::processingDataType() const
    {
        return m_processingDataType;
    }

    inline std::size_t AudioConfiguration::parametricEqFilterCount() const
    {
        return m_parametricEqFilterCount;
    }

    inline const std::vector<double>& AudioConfiguration::eqCenterFrequencies() const
    {
        return m_eqCenterFrequencies;
    }

    inline std::size_t AudioConfiguration::soundLevelLength() const
    {
        return m_soundLevelLength;
    }
}

#endif
