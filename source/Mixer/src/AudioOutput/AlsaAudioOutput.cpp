#include <Mixer/AudioOutput/AlsaAudioOutput.h>

using namespace adaptone;
using namespace std;

AlsaAudioOutput::AlsaAudioOutput(PcmAudioFrame::Format format,
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
