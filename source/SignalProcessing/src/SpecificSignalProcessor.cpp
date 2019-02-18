#include <SignalProcessing/SpecificSignalProcessor.h>

using namespace adaptone;

SpecificSignalProcessor::SpecificSignalProcessor()
{
}

SpecificSignalProcessor::~SpecificSignalProcessor()
{
}

const PcmAudioFrame& SpecificSignalProcessor::process(const PcmAudioFrame& inputFrame)
{
    return inputFrame;
}
