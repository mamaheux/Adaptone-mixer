#include <Uniformization/Communication/Messages/Tcp/FftRequestMessage.h>

using namespace adaptone;

constexpr uint32_t FftRequestMessage::Id;
constexpr size_t FftRequestMessage::MessageSize;

static constexpr size_t HoursFromBufferOffset = 8;
static constexpr size_t MinutesFromBufferOffset = 9;
static constexpr size_t SecondsFromBufferOffset = 10;
static constexpr size_t MillisecondsFromBufferOffset = 11;
static constexpr size_t FftIdFromBufferOffset = 13;

static constexpr size_t HoursSerializePayloadOffset = 0;
static constexpr size_t MinutesSerializePayloadOffset = 1;
static constexpr size_t SecondsSerializePayloadOffset = 2;
static constexpr size_t MillisecondsSerializePayloadOffset = 3;
static constexpr size_t FftIdSerializePayloadOffset = 5;


FftRequestMessage::FftRequestMessage(uint8_t hours, uint8_t minutes, uint8_t seconds,
    uint16_t milliseconds, uint16_t fftId) :
    PayloadMessage(Id, 7),
    m_hours(hours),
    m_minutes(minutes),
    m_seconds(seconds),
    m_milliseconds(milliseconds),
    m_fftId(fftId)
{
}

FftRequestMessage::~FftRequestMessage()
{
}

FftRequestMessage FftRequestMessage::fromBuffer(NetworkBufferView buffer, size_t messageSize)
{
    verifyId(buffer, Id);
    verifyMessageSize(messageSize, MessageSize);

    return FftRequestMessage(buffer.data()[HoursFromBufferOffset],
        buffer.data()[MinutesFromBufferOffset],
        buffer.data()[SecondsFromBufferOffset],
        boost::endian::big_to_native(*reinterpret_cast<uint16_t*>(buffer.data() + MillisecondsFromBufferOffset)),
        boost::endian::big_to_native(*reinterpret_cast<uint16_t*>(buffer.data() + FftIdFromBufferOffset)));
}

void FftRequestMessage::serializePayload(NetworkBufferView buffer) const
{
    buffer.data()[HoursSerializePayloadOffset] = m_hours;
    buffer.data()[MinutesSerializePayloadOffset] = m_minutes;
    buffer.data()[SecondsSerializePayloadOffset] = m_seconds;

    *reinterpret_cast<uint16_t*>(buffer.data() + MillisecondsSerializePayloadOffset) =
        boost::endian::native_to_big(m_milliseconds);
    *reinterpret_cast<uint16_t*>(buffer.data() + FftIdSerializePayloadOffset) = boost::endian::native_to_big(m_fftId);
}
