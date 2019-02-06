#include <Mixer/AudioOutput/AudioOutput.h>

#include <Utils/Exception/NotSupportedException.h>

using namespace adaptone;
using namespace std;

AudioOutput::AudioOutput(PcmAudioFrame::Format format, std::size_t channelCount, std::size_t frameSampleCount) :
    m_format(format), m_channelCount(channelCount), m_frameSampleCount(frameSampleCount)
{
}

AudioOutput::~AudioOutput()
{
}

bool AudioOutput::hasGainControl()
{
    return false;
}

void AudioOutput::setGain(std::size_t channelIndex, uint8_t gain)
{
    THROW_NOT_SUPPORTED_EXCEPTION("");
}
