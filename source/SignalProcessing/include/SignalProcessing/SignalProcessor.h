#ifndef SIGNAL_PROCESSING_SIGNAL_PROCESSOR_H
#define SIGNAL_PROCESSING_SIGNAL_PROCESSOR_H

#include <SignalProcessing/ProcessingDataType.h>
#include <SignalProcessing/SpecificSignalProcessor.h>

#include <Utils/ClassMacro.h>
#include <Utils/Data/PcmAudioFrame.h>

#include <memory>

namespace adaptone
{
    class SignalProcessor
    {
        std::unique_ptr<SpecificSignalProcessor> m_specificSignalProcessor;

    public:
        SignalProcessor(ProcessingDataType processingDataType,
            std::size_t frameSampleCount,
            std::size_t sampleFrequency,
            std::size_t inputChannelCount,
            std::size_t outputChannelCount,
            PcmAudioFrame::Format inputFormat,
            PcmAudioFrame::Format outputFormat);
        virtual ~SignalProcessor();

        DECLARE_NOT_COPYABLE(SignalProcessor);
        DECLARE_NOT_MOVABLE(SignalProcessor);

        const PcmAudioFrame& process(const PcmAudioFrame& inputFrame);
    };

    inline const PcmAudioFrame& SignalProcessor::process(const PcmAudioFrame& inputFrame)
    {
        return m_specificSignalProcessor->process(inputFrame);
    }

}

#endif
