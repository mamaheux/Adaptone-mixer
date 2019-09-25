#if defined(__unix__) || defined(__linux__)

#include <Mixer/AudioOutput/AlsaAudioOutput.h>

using namespace adaptone;
using namespace std;

AlsaAudioOutput::AlsaAudioOutput(PcmAudioFrameFormat format,
    size_t channelCount,
    size_t frameSampleCount,
    size_t sampleFrequency,
    const string& device) :
    AudioOutput(format, channelCount, frameSampleCount),
    m_device(device,
        AlsaPcmDevice::Stream::Playback,
        format,
        channelCount,
        frameSampleCount,
        sampleFrequency)
{
}

AlsaAudioOutput::~AlsaAudioOutput()
{
}

void AlsaAudioOutput::write(const PcmAudioFrame& frame)
{
    m_device.write(frame);
}

#endif
