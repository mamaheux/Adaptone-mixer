#include <Utils/Data/PcmAudioFrame.h>

#include <Utils/Exception/InvalidValueException.h>

using namespace adaptone;
using namespace std;

PcmAudioFrame::Format PcmAudioFrame::parseFormat(const std::string& format)
{
    if (format == "signed_8")
    {
        return PcmAudioFrame::Format::Signed8;
    }
    if (format == "signed_16")
    {
        return PcmAudioFrame::Format::Signed16;
    }
    if (format == "signed_24")
    {
        return PcmAudioFrame::Format::Signed24;
    }
    if (format == "signed_padded_24")
    {
        return PcmAudioFrame::Format::SignedPadded24;
    }
    if (format == "signed_32")
    {
        return PcmAudioFrame::Format::Signed32;
    }
    if (format == "unsigned_8")
    {
        return PcmAudioFrame::Format::Unsigned8;
    }
    if (format == "unsigned_16")
    {
        return PcmAudioFrame::Format::Unsigned16;
    }
    if (format == "unsigned_24")
    {
        return PcmAudioFrame::Format::Unsigned24;
    }
    if (format == "unsigned_padded_24")
    {
        return PcmAudioFrame::Format::UnsignedPadded24;
    }
    if (format == "unsigned_32")
    {
        return PcmAudioFrame::Format::Unsigned32;
    }
    THROW_INVALID_VALUE_EXCEPTION("PcmAudioFrame::Format", format);
}

PcmAudioFrame::PcmAudioFrame(Format format, size_t channelCount, size_t sampleCount) :
    m_format(format), m_channelCount(channelCount), m_sampleCount(sampleCount)
{
    m_data = new uint8_t[size()];
}

PcmAudioFrame::PcmAudioFrame(const PcmAudioFrame& other) :
    m_format(other.m_format), m_channelCount(other.m_channelCount), m_sampleCount(other.m_sampleCount)
{
    m_data = new uint8_t[size()];
    memcpy(m_data, other.m_data, size());
}

PcmAudioFrame::PcmAudioFrame(PcmAudioFrame&& other) :
    m_format(other.m_format), m_channelCount(other.m_channelCount), m_sampleCount(other.m_sampleCount)
{
    m_data = other.m_data;

    other.m_channelCount = 0;
    other.m_sampleCount = 0;
    other.m_data = nullptr;
}

PcmAudioFrame::~PcmAudioFrame()
{
    if (m_data != nullptr)
    {
        delete[] m_data;
    }
}

PcmAudioFrame& PcmAudioFrame::operator=(const PcmAudioFrame& other)
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

PcmAudioFrame& PcmAudioFrame::operator=(PcmAudioFrame&& other)
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
