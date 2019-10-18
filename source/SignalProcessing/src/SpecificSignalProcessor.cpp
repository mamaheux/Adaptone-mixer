#include <SignalProcessing/SpecificSignalProcessor.h>

using namespace adaptone;
using namespace std;

SpecificSignalProcessor::SpecificSignalProcessor()
{
}

SpecificSignalProcessor::~SpecificSignalProcessor()
{
}

void SpecificSignalProcessor::setInputGain(size_t channelIndex, double gain)
{
}

void SpecificSignalProcessor::setInputGains(const vector<double>& gains)
{
}

void SpecificSignalProcessor::setInputGraphicEqGains(size_t channelIndex, const vector<double>& gains)
{
}

void SpecificSignalProcessor::setMixingGain(size_t inputChannelIndex, size_t outputChannelIndex, double gain)
{
}

void SpecificSignalProcessor::setMixingGains(size_t outputChannelIndex, const vector<double>& gains)
{
}

void SpecificSignalProcessor::setMixingGains(const vector<double>& gains)
{
}

void SpecificSignalProcessor::setOutputGraphicEqGains(size_t channelIndex, const vector<double>& gains)
{
}

void SpecificSignalProcessor::setOutputGraphicEqGains(size_t startChannelIndex, size_t n, const vector<double>& gains)
{
}

void SpecificSignalProcessor::setUniformizationGraphicEqGains(std::size_t channelIndex,
    const vector<double>& gains)
{
}

void SpecificSignalProcessor::setUniformizationGraphicEqGains(std::size_t startChannelIndex, std::size_t n,
    const vector<double>& gains)
{
}

void SpecificSignalProcessor::setOutputGain(size_t channelIndex, double gain)
{
}

void SpecificSignalProcessor::setOutputGains(const vector<double>& gains)
{
}

void SpecificSignalProcessor::setOutputDelay(size_t channelIndex, size_t delay)
{
}

void SpecificSignalProcessor::setOutputDelays(const vector<size_t>& delays)
{
}

const PcmAudioFrame& SpecificSignalProcessor::process(const PcmAudioFrame& inputFrame)
{
    return inputFrame;
}
