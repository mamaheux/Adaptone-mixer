#include <Mixer/AudioOutput/AudioOutput.h>

#include <Utils/Exception/NotSupportedException.h>

using namespace adaptone;
using namespace std;

AudioOutput::AudioOutput(PcmAudioFrameFormat format, size_t channelCount, size_t frameSampleCount) :
    m_format(format), m_channelCount(channelCount), m_frameSampleCount(frameSampleCount)
{
}

AudioOutput::~AudioOutput()
{
}
