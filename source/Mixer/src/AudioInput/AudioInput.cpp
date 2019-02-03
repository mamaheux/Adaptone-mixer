#include <Mixer/AudioInput/AudioInput.h>

#include <Utils/Exception/NotSupportedException.h>

using namespace adaptone;
using namespace std;

AudioInput::AudioInput(PcmAudioFrame::Format format, size_t channelCount, size_t frameSampleCount) :
    m_frame(format, channelCount, frameSampleCount)
{
}

AudioInput::~AudioInput()
{
}

bool AudioInput::hasGainControl()
{
    return false;
}

void AudioInput::setGain(std::size_t channelIndex, uint8_t gain)
{
    THROW_NOT_SUPPORTED_EXCEPTION("");
}