#include <Mixer/AudioOutput/RawFileAudioOutput.h>

#include <Utils/Exception/InvalidValueException.h>

using namespace adaptone;
using namespace std;

RawFileAudioOutput::RawFileAudioOutput(PcmAudioFrame::Format format,
    size_t channelCount,
    size_t frameSampleCount,
    const string& filename) :
    AudioOutput(format, channelCount, frameSampleCount),
    m_fileStream(filename, ofstream::binary)
{
}

RawFileAudioOutput::~RawFileAudioOutput()
{
}

void RawFileAudioOutput::write(const PcmAudioFrame& frame)
{
    if (frame.format() != m_format ||
        frame.channelCount() != m_channelCount ||
        frame.sampleCount() != m_frameSampleCount)
    {
        THROW_INVALID_VALUE_EXCEPTION("format, channelCount, sampleCount", "");
    }

    m_fileStream << frame;
}
