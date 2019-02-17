#include <SignalProcessing/SignalProcessor.h>

#include <SignalProcessing/Cuda/CudaSignalProcessor.h>

#include <Utils/Exception/NotSupportedException.h>

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
    if (processingDataType == ProcessingDataType::Float)
    {
        m_specificSignalProcessor = make_unique<CudaSignalProcessor<float>>(frameSampleCount,
            sampleFrequency,
            inputChannelCount,
            outputChannelCount,
            inputFormat,
            outputFormat);
    }
    else if (processingDataType == ProcessingDataType::Double)
    {
        m_specificSignalProcessor = make_unique<CudaSignalProcessor<double>>(frameSampleCount,
            sampleFrequency,
            inputChannelCount,
            outputChannelCount,
            inputFormat,
            outputFormat);
    }
    else
    {
        THROW_NOT_SUPPORTED_EXCEPTION("Not supported processing data type");
    }
}

SignalProcessor::~SignalProcessor()
{
}
