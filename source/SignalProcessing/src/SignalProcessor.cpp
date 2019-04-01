#include <SignalProcessing/SignalProcessor.h>

#ifndef USE_CUDA

using namespace adaptone;
using namespace std;

SignalProcessor::SignalProcessor(ProcessingDataType processingDataType,
    size_t frameSampleCount,
    size_t sampleFrequency,
    size_t inputChannelCount,
    size_t outputChannelCount,
    PcmAudioFrame::Format inputFormat,
    PcmAudioFrame::Format outputFormat,
    std::size_t eqParametricFilterCount,
    const std::vector<double>& eqCenterFrequencies)
{
    m_specificSignalProcessor = make_unique<SpecificSignalProcessor>();
}

SignalProcessor::~SignalProcessor()
{
}

#endif
