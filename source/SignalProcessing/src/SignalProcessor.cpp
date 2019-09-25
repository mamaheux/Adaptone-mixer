#include <SignalProcessing/SignalProcessor.h>

#ifndef USE_CUDA

using namespace adaptone;
using namespace std;

SignalProcessor::SignalProcessor(ProcessingDataType processingDataType,
    size_t frameSampleCount,
    size_t sampleFrequency,
    size_t inputChannelCount,
    size_t outputChannelCount,
    PcmAudioFrameFormat inputFormat,
    PcmAudioFrameFormat outputFormat,
    const vector<double>& eqCenterFrequencies,
    size_t maxOutputDelay,
    size_t soundLevelLength,
    shared_ptr<AnalysisDispatcher> analysisDispatcher)
{
    m_specificSignalProcessor = make_unique<SpecificSignalProcessor>();
}

SignalProcessor::~SignalProcessor()
{
}

#endif
