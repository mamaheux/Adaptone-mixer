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
