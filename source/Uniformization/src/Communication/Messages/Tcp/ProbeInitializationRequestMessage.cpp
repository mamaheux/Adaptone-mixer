#include <Uniformization/Communication/Messages/Tcp/ProbeInitializationRequestMessage.h>

#include <Utils/Exception/InvalidValueException.h>

#include <unordered_map>

using namespace adaptone;
using namespace std;

constexpr uint32_t ProbeInitializationRequestMessage::Id;

ProbeInitializationRequestMessage::ProbeInitializationRequestMessage(uint32_t sampleFrequency,
    PcmAudioFrame::Format format) :
    PayloadMessage(Id, 2 * sizeof(uint32_t)),
    m_sampleFrequency(sampleFrequency),
    m_format(format)
{
}

ProbeInitializationRequestMessage::~ProbeInitializationRequestMessage()
{
}

void ProbeInitializationRequestMessage::serializePayload(NetworkBufferView& buffer)
{
    uint32_t* data = reinterpret_cast<uint32_t*>(buffer.data());
    data[0] = boost::endian::native_to_big(m_sampleFrequency);
    data[1] = boost::endian::native_to_big(serializeFormat(m_format));
}

uint32_t ProbeInitializationRequestMessage::serializeFormat(PcmAudioFrame::Format format)
{
    const unordered_map<PcmAudioFrame::Format, uint32_t> Mapping(
        {
            { PcmAudioFrame::Format::Signed8, 0 },
            { PcmAudioFrame::Format::Signed16, 1 },
            { PcmAudioFrame::Format::Signed24, 2 },
            { PcmAudioFrame::Format::SignedPadded24, 3 },
            { PcmAudioFrame::Format::Signed32, 4 },

            { PcmAudioFrame::Format::Unsigned8, 5 },
            { PcmAudioFrame::Format::Unsigned16, 6 },
            { PcmAudioFrame::Format::Unsigned24, 7 },
            { PcmAudioFrame::Format::UnsignedPadded24, 8 },
            { PcmAudioFrame::Format::Unsigned32, 9 },

            { PcmAudioFrame::Format::Float, 10 },
            { PcmAudioFrame::Format::Double, 11 }
        });

    auto it = Mapping.find(format);
    if (it != Mapping.end())
    {
        return it->second;
    }

    THROW_INVALID_VALUE_EXCEPTION("Not supported format", "");
}

PcmAudioFrame::Format ProbeInitializationRequestMessage::parseFormat(uint32_t format)
{
    const unordered_map<uint32_t, PcmAudioFrame::Format> Mapping(
        {
            { 0, PcmAudioFrame::Format::Signed8 },
            { 1, PcmAudioFrame::Format::Signed16 },
            { 2, PcmAudioFrame::Format::Signed24 },
            { 3, PcmAudioFrame::Format::SignedPadded24 },
            { 4, PcmAudioFrame::Format::Signed32 },

            { 5, PcmAudioFrame::Format::Unsigned8 },
            { 6, PcmAudioFrame::Format::Unsigned16 },
            { 7, PcmAudioFrame::Format::Unsigned24 },
            { 8, PcmAudioFrame::Format::UnsignedPadded24 },
            { 9, PcmAudioFrame::Format::Unsigned32 },

            { 10, PcmAudioFrame::Format::Float },
            { 11, PcmAudioFrame::Format::Double }
        });

    auto it = Mapping.find(format);
    if (it != Mapping.end())
    {
        return it->second;
    }

    THROW_INVALID_VALUE_EXCEPTION("Not supported format", "");
}
