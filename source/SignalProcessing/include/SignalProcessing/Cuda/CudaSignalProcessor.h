#ifndef SIGNAL_PROCESSING_CUDA_CUDA_SIGNAL_PROCESSOR_H
#define SIGNAL_PROCESSING_CUDA_CUDA_SIGNAL_PROCESSOR_H

#include <SignalProcessing/ProcessingDataType.h>

#include <Utils/Data/PcmAudioFrame.h>

namespace adaptone
{
    class CudaSignalProcessor
    {
        ProcessingDataType m_processingDataType;

        std::size_t m_frameSampleCount;
        std::size_t m_sampleFrequency;

        std::size_t m_inputChannelCount;
        std::size_t m_outputChannelCount;

        PcmAudioFrame::Format m_inputFormat;
        PcmAudioFrame::Format m_outputFormat;

    public:
        CudaSignalProcessor(ProcessingDataType processingDataType,
            std::size_t frameSampleCount,
            std::size_t sampleFrequency,
            std::size_t inputChannelCount,
            std::size_t outputChannelCount,
            PcmAudioFrame::Format inputFormat,
            PcmAudioFrame::Format outputFormat);
        virtual ~CudaSignalProcessor();

        const PcmAudioFrame& process(const PcmAudioFrame& inputFrame);
    };
}

#endif
