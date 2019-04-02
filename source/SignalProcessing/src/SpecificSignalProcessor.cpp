#include <SignalProcessing/SpecificSignalProcessor.h>

using namespace adaptone;
using namespace std;

SpecificSignalProcessor::SpecificSignalProcessor()
{
}

SpecificSignalProcessor::~SpecificSignalProcessor()
{
}

void SpecificSignalProcessor::setInputGain(size_t channel, double gainDb)
{
}

void SpecificSignalProcessor::setInputGains(const vector<double>& gainsDb)
{
}

void SpecificSignalProcessor::setInputParametricEqParameters(std::size_t channel,
    const std::vector<ParametricEqParameters>& parameters)
{
}

void SpecificSignalProcessor::setInputGraphicEqGains(std::size_t channel, const std::vector<double>& gainsDb)
{
}

void SpecificSignalProcessor::setMixingGain(size_t inputChannel, size_t outputChannel, double gainDb)
{
}

void SpecificSignalProcessor::setMixingGains(size_t outputChannel, const vector<double>& gainsDb)
{
}

void SpecificSignalProcessor::setMixingGains(const vector<double>& gainsDb)
{
}

void SpecificSignalProcessor::setOutputParametricEqParameters(std::size_t channel,
    const std::vector<ParametricEqParameters>& parameters)
{
}

void SpecificSignalProcessor::setOutputGraphicEqGains(std::size_t channel, const std::vector<double>& gainsDb)
{
}

void SpecificSignalProcessor::setOutputGain(size_t channel, double gainDb)
{
}

void SpecificSignalProcessor::setOutputGains(const vector<double>& gainsDb)
{
}

const PcmAudioFrame& SpecificSignalProcessor::process(const PcmAudioFrame& inputFrame)
{
    return inputFrame;
}
