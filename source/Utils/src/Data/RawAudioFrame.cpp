#include <Utils/Data/RawAudioFrame.h>

#include <Utils/Exception/InvalidValueException.h>

using namespace adaptone;
using namespace std;

RawAudioFrame::Format RawAudioFrame::parseFormat(const std::string& format)
{
    if (format == "signed_8")
    {
        return RawAudioFrame::Format::Signed8;
    }
    if (format == "signed_16")
    {
        return RawAudioFrame::Format::Signed16;
    }
    if (format == "signed_24")
    {
        return RawAudioFrame::Format::Signed24;
    }
    if (format == "signed_padded_24")
    {
        return RawAudioFrame::Format::SignedPadded24;
    }
    if (format == "signed_32")
    {
        return RawAudioFrame::Format::Signed32;
    }
    if (format == "unsigned_8")
    {
        return RawAudioFrame::Format::Unsigned8;
    }
    if (format == "unsigned_16")
    {
        return RawAudioFrame::Format::Unsigned16;
    }
    if (format == "unsigned_24")
    {
        return RawAudioFrame::Format::Unsigned24;
    }
    if (format == "unsigned_padded_24")
    {
        return RawAudioFrame::Format::UnsignedPadded24;
    }
    if (format == "unsigned_32")
    {
        return RawAudioFrame::Format::Unsigned32;
    }
    THROW_INVALID_VALUE_EXCEPTION("RawAudioFrame::Format", format);
}

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