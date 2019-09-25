#if defined(__unix__) || defined(__linux__)

#include <Mixer/AudioInput/AlsaAudioInput.h>

using namespace adaptone;
using namespace std;

AlsaAudioInput::AlsaAudioInput(PcmAudioFrameFormat format,
    size_t channelCount,
    size_t frameSampleCount,
    size_t sampleFrequency,
    const string& device) :
    AudioInput(format, channelCount, frameSampleCount),
    m_device(device,
        AlsaPcmDevice::Stream::Capture,
        format,
        channelCount,
        frameSampleCount,
        sampleFrequency)
{
}

AlsaAudioInput::~AlsaAudioInput()
{
}

const PcmAudioFrame& AlsaAudioInput::read()
{
    while (!m_device.read(m_frame))
    {
    }

    return m_frame;
}

bool AlsaAudioInput::hasNext()
{
    return true;
}

#endif
