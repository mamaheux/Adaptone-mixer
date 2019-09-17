#include <Utils/Data/PcmAudioFrame.h>

#include <Utils/Exception/InvalidValueException.h>

#include <unordered_map>

using namespace adaptone;
using namespace std;

PcmAudioFrame::Format PcmAudioFrame::parseFormat(const string& format)
{
    const unordered_map<string, PcmAudioFrame::Format> Mapping(
        {
            { "signed_8", PcmAudioFrame::Format::Signed8 },
            { "signed_16", PcmAudioFrame::Format::Signed16 },
            { "signed_24", PcmAudioFrame::Format::Signed24 },
            { "signed_padded_24", PcmAudioFrame::Format::SignedPadded24 },
            { "signed_32", PcmAudioFrame::Format::Signed32 },

            { "unsigned_8", PcmAudioFrame::Format::Unsigned8 },
            { "unsigned_16", PcmAudioFrame::Format::Unsigned16 },
            { "unsigned_24", PcmAudioFrame::Format::Unsigned24 },
            { "unsigned_padded_24", PcmAudioFrame::Format::UnsignedPadded24 },
            { "unsigned_32", PcmAudioFrame::Format::Unsigned32 },

            { "float", PcmAudioFrame::Format::Float },
            { "double", PcmAudioFrame::Format::Double }
        });

    auto it = Mapping.find(format);
    if (it != Mapping.end())
    {
        return it->second;
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
    if (m_format != other.m_format || m_channelCount != other.m_channelCount || m_sampleCount != other.m_sampleCount)
    {
        if (m_data != nullptr)
        {
            delete[] m_data;
        }

        m_format = other.m_format;
        m_channelCount = other.m_channelCount;
        m_sampleCount = other.m_sampleCount;

        m_data = new uint8_t[size()];
    }
    memcpy(m_data, other.m_data, size());

    return *this;
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

    return *this;
}

void PcmAudioFrame::writeChannel(size_t thisChannelIndex, const PcmAudioFrame& other, size_t otherChannelIndex)
{
    if (other.m_format != m_format || other.m_sampleCount != m_sampleCount)
    {
        THROW_INVALID_VALUE_EXCEPTION("format, sampleCount", "");
    }

    size_t sampleSize = formatSize(m_format);
    for (size_t i = 0; i < other.m_sampleCount; i++)
    {
        size_t thisDataIndex = (i * m_channelCount + thisChannelIndex) * sampleSize;
        size_t otherDataIndex = (i * other.m_channelCount + otherChannelIndex) * sampleSize;

        for (size_t j = 0; j < sampleSize; j++)
        {
            m_data[thisDataIndex + j] = other.m_data[otherDataIndex + j];
        }
    }
}
