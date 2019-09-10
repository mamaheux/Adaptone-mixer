#include <SignalProcessing/SpecificSignalProcessor.h>

using namespace adaptone;
using namespace std;

SpecificSignalProcessor::SpecificSignalProcessor()
{
}

SpecificSignalProcessor::~SpecificSignalProcessor()
{
}

void SpecificSignalProcessor::setInputGain(size_t channel, double gain)
{
}

void SpecificSignalProcessor::setInputGains(const vector<double>& gains)
{
}

void SpecificSignalProcessor::setInputGraphicEqGains(size_t channel, const vector<double>& gains)
{
}

void SpecificSignalProcessor::setMixingGain(size_t inputChannel, size_t outputChannel, double gain)
{
}

void SpecificSignalProcessor::setMixingGains(size_t outputChannel, const vector<double>& gains)
{
}

void SpecificSignalProcessor::setMixingGains(const vector<double>& gains)
{
}

void SpecificSignalProcessor::setOutputGraphicEqGains(size_t channel, const vector<double>& gains)
{
}

void SpecificSignalProcessor::setOutputGraphicEqGains(size_t startChannelIndex, size_t n, const vector<double>& gains)
{
}

void SpecificSignalProcessor::setOutputGain(size_t channel, double gain)
{
}

void SpecificSignalProcessor::setOutputGains(const vector<double>& gains)
{
}

void SpecificSignalProcessor::setOutputDelay(size_t channel, size_t delay)
{
}

void SpecificSignalProcessor::setOutputDelays(const vector<size_t>& delays)
{
}

const PcmAudioFrame& SpecificSignalProcessor::process(const PcmAudioFrame& inputFrame)
{
    return inputFrame;
}
