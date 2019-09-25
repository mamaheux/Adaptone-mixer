#include <Uniformization/Communication/Messages/Tcp/ProbeInitializationRequestMessage.h>

#include <Utils/Exception/InvalidValueException.h>

#include <unordered_map>

using namespace adaptone;
using namespace std;

constexpr uint32_t ProbeInitializationRequestMessage::Id;
constexpr size_t ProbeInitializationRequestMessage::MessageSize;

ProbeInitializationRequestMessage::ProbeInitializationRequestMessage(uint32_t sampleFrequency,
    PcmAudioFrameFormat format) :
    PayloadMessage(Id, 2 * sizeof(uint32_t)),
    m_sampleFrequency(sampleFrequency),
    m_format(format)
{
}

ProbeInitializationRequestMessage::~ProbeInitializationRequestMessage()
{
}

ProbeInitializationRequestMessage ProbeInitializationRequestMessage::fromBuffer(NetworkBufferView buffer,
    size_t messageSize)
{
    verifyId(buffer, Id);
    verifyMessageSize(messageSize, MessageSize);

    uint32_t* data = reinterpret_cast<uint32_t*>(buffer.data() + 8);
    return ProbeInitializationRequestMessage(boost::endian::big_to_native(data[0]),
        parseFormat(boost::endian::big_to_native(data[1])));
}

void ProbeInitializationRequestMessage::serializePayload(NetworkBufferView buffer) const
{
    uint32_t* data = reinterpret_cast<uint32_t*>(buffer.data());
    data[0] = boost::endian::native_to_big(m_sampleFrequency);
    data[1] = boost::endian::native_to_big(serializeFormat(m_format));
}

uint32_t ProbeInitializationRequestMessage::serializeFormat(PcmAudioFrameFormat format)
{
    static const unordered_map<PcmAudioFrameFormat, uint32_t> Mapping(
        {
            { PcmAudioFrameFormat::Signed8, 0 },
            { PcmAudioFrameFormat::Signed16, 1 },
            { PcmAudioFrameFormat::Signed24, 2 },
            { PcmAudioFrameFormat::SignedPadded24, 3 },
            { PcmAudioFrameFormat::Signed32, 4 },

            { PcmAudioFrameFormat::Unsigned8, 5 },
            { PcmAudioFrameFormat::Unsigned16, 6 },
            { PcmAudioFrameFormat::Unsigned24, 7 },
            { PcmAudioFrameFormat::UnsignedPadded24, 8 },
            { PcmAudioFrameFormat::Unsigned32, 9 },

            { PcmAudioFrameFormat::Float, 10 },
            { PcmAudioFrameFormat::Double, 11 }
        });

    auto it = Mapping.find(format);
    if (it != Mapping.end())
    {
        return it->second;
    }

    THROW_INVALID_VALUE_EXCEPTION("Format not supported", "");
}

PcmAudioFrameFormat ProbeInitializationRequestMessage::parseFormat(uint32_t format)
{
    static const unordered_map<uint32_t, PcmAudioFrameFormat> Mapping(
        {
            { 0, PcmAudioFrameFormat::Signed8 },
            { 1, PcmAudioFrameFormat::Signed16 },
            { 2, PcmAudioFrameFormat::Signed24 },
            { 3, PcmAudioFrameFormat::SignedPadded24 },
            { 4, PcmAudioFrameFormat::Signed32 },

            { 5, PcmAudioFrameFormat::Unsigned8 },
            { 6, PcmAudioFrameFormat::Unsigned16 },
            { 7, PcmAudioFrameFormat::Unsigned24 },
            { 8, PcmAudioFrameFormat::UnsignedPadded24 },
            { 9, PcmAudioFrameFormat::Unsigned32 },

            { 10, PcmAudioFrameFormat::Float },
            { 11, PcmAudioFrameFormat::Double }
        });

    auto it = Mapping.find(format);
    if (it != Mapping.end())
    {
        return it->second;
    }

    THROW_INVALID_VALUE_EXCEPTION("Format not supported", "");
}
