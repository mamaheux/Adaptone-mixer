#include <SignalProcessing/SignalProcessor.h>

using namespace adaptone;
using namespace std;

SignalProcessor::SignalProcessor(ProcessingDataType processingDataType,
    size_t frameSampleCount,
    size_t sampleFrequency,
    size_t inputChannelCount,
    size_t outputChannelCount,
    PcmAudioFrame::Format inputFormat,
    PcmAudioFrame::Format outputFormat)
{
#ifdef USE_CUDA

    m_cudaSignalProcessor = make_unique<CudaSignalProcessor>(processingDataType,
        frameSampleCount,
        sampleFrequency,
        inputChannelCount,
        outputChannelCount,
        inputFormat,
        outputFormat);

#endif
}

SignalProcessor::~SignalProcessor()
{
}

const PcmAudioFrame& SignalProcessor::process(const PcmAudioFrame& inputFrame)
{
#ifdef USE_CUDA

    return m_cudaSignalProcessor->process(inputFrame);

#else

    return inputFrame;

#endif
}
