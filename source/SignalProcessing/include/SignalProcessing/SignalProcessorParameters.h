#ifndef SIGNAL_PROCESSING_SIGNAL_PROCESSOR_PARAMETERS_H
#define SIGNAL_PROCESSING_SIGNAL_PROCESSOR_PARAMETERS_H

#include <SignalProcessing/ProcessingDataType.h>

#include <Utils/Data/PcmAudioFrameFormat.h>

#include <cstddef>
#include <vector>

namespace adaptone
{
    class SignalProcessorParameters
    {
        ProcessingDataType m_processingDataType;
        std::size_t m_frameSampleCount;
        std::size_t m_sampleFrequency;
        std::size_t m_inputChannelCount;
        std::size_t m_outputChannelCount;
        PcmAudioFrameFormat m_inputFormat;
        PcmAudioFrameFormat m_outputFormat;
        std::vector<double> m_eqCenterFrequencies;
        std::size_t m_maxOutputDelay;
        std::size_t m_soundLevelLength;

    public:
        SignalProcessorParameters(ProcessingDataType processingDataType,
            std::size_t frameSampleCount,
            std::size_t sampleFrequency,
            std::size_t inputChannelCount,
            std::size_t outputChannelCount,
            PcmAudioFrameFormat inputFormat,
            PcmAudioFrameFormat outputFormat,
            std::vector<double> eqCenterFrequencies,
            std::size_t maxOutputDelay,
            std::size_t soundLevelLength);
        virtual ~SignalProcessorParameters();

        ProcessingDataType processingDataType() const;
        std::size_t frameSampleCount() const;
        std::size_t sampleFrequency() const;
        std::size_t inputChannelCount() const;
        std::size_t outputChannelCount() const;
        PcmAudioFrameFormat inputFormat() const;
        PcmAudioFrameFormat outputFormat() const;
        const std::vector<double>& eqCenterFrequencies() const;
        std::size_t maxOutputDelay() const;
        std::size_t soundLevelLength() const;
    };

    inline ProcessingDataType SignalProcessorParameters::processingDataType() const
    {
        return m_processingDataType;
    }

    inline std::size_t SignalProcessorParameters::frameSampleCount() const
    {
        return m_frameSampleCount;
    }

    inline std::size_t SignalProcessorParameters::sampleFrequency() const
    {
        return m_sampleFrequency;
    }

    inline std::size_t SignalProcessorParameters::inputChannelCount() const
    {
        return m_inputChannelCount;
    }

    inline std::size_t SignalProcessorParameters::outputChannelCount() const
    {
        return m_outputChannelCount;
    }

    inline PcmAudioFrameFormat SignalProcessorParameters::inputFormat() const
    {
        return m_inputFormat;
    }

    inline PcmAudioFrameFormat SignalProcessorParameters::outputFormat() const
    {
        return m_outputFormat;
    }

    inline const std::vector<double>& SignalProcessorParameters::eqCenterFrequencies() const
    {
        return m_eqCenterFrequencies;
    }

    inline std::size_t SignalProcessorParameters::maxOutputDelay() const
    {
        return m_maxOutputDelay;
    }

    inline std::size_t SignalProcessorParameters::soundLevelLength() const
    {
        return m_soundLevelLength;
    }
}

#endif
