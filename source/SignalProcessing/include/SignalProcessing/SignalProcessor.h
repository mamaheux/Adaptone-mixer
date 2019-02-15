#ifndef SIGNAL_PROCESSING_SIGNAL_PROCESSOR_H
#define SIGNAL_PROCESSING_SIGNAL_PROCESSOR_H

#include <Utils/Data/PcmAudioFrame.h>

namespace adaptone
{
    class SignalProcessor
    {
    public:
        enum class ProcessingDataType
        {
            Float,
            Double
        };

    private:
        ProcessingDataType m_processingDataType;

        std::size_t m_frameSampleCount;
        std::size_t m_sampleFrequency;

        std::size_t m_inputChannelCount;
        std::size_t m_outputChannelCount;

        PcmAudioFrame::Format m_inputFormat;
        PcmAudioFrame::Format m_outputFormat;

    public:
        SignalProcessor(ProcessingDataType processingDataType,
            std::size_t frameSampleCount,
            std::size_t sampleFrequency,
            std::size_t inputChannelCount,
            std::size_t outputChannelCount,
            PcmAudioFrame::Format inputFormat,
            PcmAudioFrame::Format outputFormat);
        virtual ~SignalProcessor();

        const PcmAudioFrame& process(const PcmAudioFrame& inputFrame);
    };
}

#endif
