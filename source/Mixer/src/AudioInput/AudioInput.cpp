#include <Mixer/AudioInput/AudioInput.h>

using namespace adaptone;
using namespace std;

AudioInput::AudioInput(PcmAudioFrame::Format format, size_t channelCount, size_t frameSampleCount) :
    m_frame(format, channelCount, frameSampleCount)
{
}

AudioInput::~AudioInput()
{
}