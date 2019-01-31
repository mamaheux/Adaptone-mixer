#include <Utils/Data/RawAudioFrame.h>

using namespace adaptone;
using namespace std;

RawAudioFrame::RawAudioFrame(Format format, size_t channelCount, size_t sampleCount) :
    m_format(format), m_channelCount(channelCount), m_sampleCount(sampleCount)
{
    m_data = new uint8_t[size()];
}

RawAudioFrame::RawAudioFrame(const RawAudioFrame& other) :
    m_format(other.m_format), m_channelCount(other.m_channelCount), m_sampleCount(other.m_sampleCount)
{
    m_data = new uint8_t[size()];
    memcpy(m_data, other.m_data, size());
}

RawAudioFrame::RawAudioFrame(RawAudioFrame&& other):
    m_format(other.m_format), m_channelCount(other.m_channelCount), m_sampleCount(other.m_sampleCount)
{
    m_data = other.m_data;

    other.m_channelCount = 0;
    other.m_sampleCount = 0;
    other.m_data = nullptr;
}

RawAudioFrame::~RawAudioFrame()
{
    if (m_data != nullptr)
    {
        delete[] m_data;
    }
}

RawAudioFrame& RawAudioFrame::operator=(const RawAudioFrame& other)
{
    if (m_data != nullptr)
    {
        delete[] m_data;
    }

    m_format = other.m_format;
    m_channelCount = other.m_channelCount;
    m_sampleCount = other.m_sampleCount;

    m_data = new uint8_t[size()];
    memcpy(m_data, other.m_data, size());
}

RawAudioFrame& RawAudioFrame::operator=(RawAudioFrame&& other)
{
    if (m_data != nullptr)
    {
        delete[] m_data;
    }

    m_format = other.m_format;
    m_channelCount = other.m_channelCount;
    m_sampleCount = other.m_sampleCount;
    m_data = other.m_data;

    other.m_channelCount = 0;
    other.m_sampleCount = 0;
    other.m_data = nullptr;
}