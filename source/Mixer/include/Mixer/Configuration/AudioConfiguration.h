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

        std::vector<double> m_eqCenterFrequencies;

        std::size_t m_maxOutputDelay; // Sample

        std::size_t m_soundLevelLength;
        std::size_t m_spectrumAnalysisFftLength;
        std::size_t m_spectrumAnalysisPointCountPerDecade;

        std::vector<std::size_t> m_headphoneChannelIndexes;

    public:
        explicit AudioConfiguration(const Properties& properties);
        virtual ~AudioConfiguration();

        std::size_t frameSampleCount() const;
        std::size_t sampleFrequency() const;
        std::size_t inputChannelCount() const;
        std::size_t outputChannelCount() const;
        ProcessingDataType processingDataType() const;

        const std::vector<double>& eqCenterFrequencies() const;

        std::size_t maxOutputDelay() const;

        std::size_t soundLevelLength() const;
        std::size_t spectrumAnalysisFftLength() const;
        std::size_t spectrumAnalysisPointCountPerDecade() const;

        std::vector<std::size_t> headphoneChannelIndexes() const;

    private:
        void setProcessingDataType(const Properties& properties);
        void setMaxOutputDelay(const Properties& properties);
        void setSpectrumAnalysisFftLength(const Properties& properties);
        void setHeadphoneChannelIndexes(const Properties& properties);
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

    inline const std::vector<double>& AudioConfiguration::eqCenterFrequencies() const
    {
        return m_eqCenterFrequencies;
    }

    inline std::size_t AudioConfiguration::maxOutputDelay() const
    {
        return m_maxOutputDelay;
    }

    inline std::size_t AudioConfiguration::soundLevelLength() const
    {
        return m_soundLevelLength;
    }

    inline std::size_t AudioConfiguration::spectrumAnalysisFftLength() const
    {
        return m_spectrumAnalysisFftLength;
    }

    inline std::size_t AudioConfiguration::spectrumAnalysisPointCountPerDecade() const
    {
        return m_spectrumAnalysisPointCountPerDecade;
    }

    inline std::vector<std::size_t> AudioConfiguration::headphoneChannelIndexes() const
    {
        return m_headphoneChannelIndexes;
    }
}

#endif
