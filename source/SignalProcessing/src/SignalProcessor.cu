#include <SignalProcessing/SignalProcessor.h>

#include <SignalProcessing/Cuda/CudaSignalProcessor.h>

#include <Utils/Exception/NotSupportedException.h>
#include <Utils/Exception/InvalidValueException.h>

using namespace adaptone;
using namespace std;

SignalProcessor::SignalProcessor(ProcessingDataType processingDataType,
    size_t frameSampleCount,
    size_t sampleFrequency,
    size_t inputChannelCount,
    size_t outputChannelCount,
    PcmAudioFrame::Format inputFormat,
    PcmAudioFrame::Format outputFormat,
    const vector<double>& eqCenterFrequencies,
    size_t maxOutputDelay,
    size_t soundLevelLength,
    shared_ptr<AnalysisDispatcher> analysisDispatcher)
{
    if (frameSampleCount < 2)
    {
        THROW_INVALID_VALUE_EXCEPTION("Invalid frame sample count value.", "< 2");
    }

    if (processingDataType == ProcessingDataType::Float)
    {
        m_specificSignalProcessor = make_unique<CudaSignalProcessor<float>>(frameSampleCount,
            sampleFrequency,
            inputChannelCount,
            outputChannelCount,
            inputFormat,
            outputFormat,
            eqCenterFrequencies,
            maxOutputDelay,
            soundLevelLength,
            analysisDispatcher);
    }
    else if (processingDataType == ProcessingDataType::Double)
    {
        m_specificSignalProcessor = make_unique<CudaSignalProcessor<double>>(frameSampleCount,
            sampleFrequency,
            inputChannelCount,
            outputChannelCount,
            inputFormat,
            outputFormat,
            eqCenterFrequencies,
            maxOutputDelay,
            soundLevelLength,
            analysisDispatcher);
    }
    else
    {
        THROW_NOT_SUPPORTED_EXCEPTION("Not supported processing data type");
    }
}

SignalProcessor::~SignalProcessor()
{
}
