#ifndef SIGNAL_PROCESSING_SIGNAL_PROCESSOR_H
#define SIGNAL_PROCESSING_SIGNAL_PROCESSOR_H

#include <SignalProcessing/ProcessingDataType.h>

#include <Utils/Data/PcmAudioFrame.h>

#include <memory>

#ifdef USE_CUDA

#include <SignalProcessing/Cuda/CudaSignalProcessor.h>

#else

class CudaSignalProcessor
{
};

#endif

namespace adaptone
{
    class SignalProcessor
    {
        std::unique_ptr<CudaSignalProcessor> m_cudaSignalProcessor;

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
